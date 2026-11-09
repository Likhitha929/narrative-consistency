import logging
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from dataset_builder import NarrativeDataset

logger = logging.getLogger(__name__)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1":       f1_score(labels, preds, average="macro", zero_division=0),
    }


# custom Trainer that uses weighted cross-entropy loss
# this is the key fix — without it the model just always predicts the majority class
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights  # tensor of shape (num_classes,)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits

        # move weights to same device as logits
        weights = self.class_weights.to(logits.device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        loss    = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


def train(
    train_samples,
    model_name="roberta-base",
    save_dir=Path("model"),
    batch_size=8,
    num_epochs=5,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    max_seq_len=256,
    label2id=None,
    id2label=None,
    seed=42,
):
    if label2id is None:
        label2id = {"consistent": 1, "contradict": 0}
    if id2label is None:
        id2label = {0: "contradict", 1: "consistent"}

    torch.manual_seed(seed)
    np.random.seed(seed)

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    logger.info(f"loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    # ── stratified 80/20 split ────────────────────────────────────────────────
    # stratified means both train and val have the same class ratio
    # this is much better than random for a small dataset
    all_labels = [s.label for s in train_samples]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(all_labels)), all_labels))

    train_data = [train_samples[i] for i in train_idx]
    val_data   = [train_samples[i] for i in val_idx]

    train_ds = NarrativeDataset(train_data, tokenizer, max_seq_len)
    val_ds   = NarrativeDataset(val_data,   tokenizer, max_seq_len)

    train_label_counts = np.bincount([s.label for s in train_data], minlength=2)
    val_label_counts   = np.bincount([s.label for s in val_data],   minlength=2)
    logger.info(f"train: {len(train_ds)} | val: {len(val_ds)}")
    logger.info(f"train labels — contradict: {train_label_counts[0]}, consistent: {train_label_counts[1]}")
    logger.info(f"val labels   — contradict: {val_label_counts[0]}, consistent: {val_label_counts[1]}")

    # ── compute class weights to fight imbalance ──────────────────────────────
    # weight = total_samples / (num_classes * count_of_class)
    # minority class (contradict) gets a higher weight → model penalised more for missing it
    total = len(train_data)
    w_contradict = total / (2 * train_label_counts[0]) if train_label_counts[0] > 0 else 1.0
    w_consistent = total / (2 * train_label_counts[1]) if train_label_counts[1] > 0 else 1.0
    class_weights = torch.tensor([w_contradict, w_consistent], dtype=torch.float)
    logger.info(f"class weights — contradict: {w_contradict:.3f}, consistent: {w_consistent:.3f}")

    total_steps  = math.ceil(len(train_ds) / batch_size) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    save_dir.mkdir(parents=True, exist_ok=True)

    use_mps = torch.backends.mps.is_available()
    logger.info(f"MPS: {use_mps} | CUDA: {torch.cuda.is_available()}")

    args = TrainingArguments(
        output_dir=str(save_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=5,
        fp16=False,
        seed=seed,
        report_to="none",
    )

    # use our custom WeightedTrainer instead of plain Trainer
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # print val report
    preds_out = trainer.predict(val_ds)
    pred_ids  = np.argmax(preds_out.predictions, axis=-1)
    true_ids  = preds_out.label_ids
    names     = [id2label[i] for i in sorted(id2label.keys())]
    logger.info(f"\n{classification_report(true_ids, pred_ids, target_names=names, zero_division=0)}")

    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    logger.info(f"model saved → {save_dir}")

    return trainer, tokenizer
