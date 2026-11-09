"""
train_nli.py

Two-stage training for our from-scratch NLI model:
  Stage 1 — train on SNLI (public 550k NLI dataset, we use 50k subset)
             so the model learns what entailment/contradiction means
  Stage 2 — fine-tune on our 388 book samples
             so the model specialises in narrative consistency

Run independently:
    python3 train_nli.py
Or run as part of main.py pipeline (replaces train_model.py).
"""

import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import AutoTokenizer

from custom_nli_model import NarrativeNLIModel

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset wrappers
# ─────────────────────────────────────────────────────────────────────────────

class SNLIDataset(Dataset):
    """Wraps SNLI samples for binary NLI (entailment=1, contradiction=0, neutral=skip)."""

    def __init__(self, texts_a, texts_b, labels, tokenizer, max_len=256):
        self.texts_a   = texts_a
        self.texts_b   = texts_b
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts_a[idx],
            self.texts_b[idx],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get("token_type_ids", torch.zeros(self.max_len, dtype=torch.long)).squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


class BookNLIDataset(Dataset):
    """Wraps our book samples (from dataset_builder)."""

    def __init__(self, samples, tokenizer, max_len=256):
        self.samples   = samples
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        # input_text format: "passage1 [SEP] passage2 [SEP] claim"
        # split into passage and claim at last [SEP]
        parts = s.input_text.rsplit(" [SEP] ", 1)
        if len(parts) == 2:
            passage, claim = parts
        else:
            passage, claim = s.input_text, ""

        enc = self.tokenizer(
            passage,
            claim,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get("token_type_ids", torch.zeros(self.max_len, dtype=torch.long)).squeeze(0),
            "labels":         torch.tensor(s.label, dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# SNLI data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_snli_binary(max_samples: int = 50000):
    """
    Downloads SNLI from HuggingFace datasets.
    We keep only entailment (→ consistent=1) and contradiction (→ contradict=0).
    Neutral examples are skipped to keep the task binary.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip3 install datasets")

    logger.info(f"loading SNLI (max {max_samples} samples per split)...")
    ds = load_dataset("snli", split="train")

    texts_a, texts_b, labels = [], [], []
    for example in ds:
        if len(texts_a) >= max_samples:
            break
        lbl = example["label"]
        if lbl == 0:   # entailment → consistent
            texts_a.append(example["premise"])
            texts_b.append(example["hypothesis"])
            labels.append(1)
        elif lbl == 2:  # contradiction → contradict
            texts_a.append(example["premise"])
            texts_b.append(example["hypothesis"])
            labels.append(0)
        # skip neutral (lbl == 1)

    logger.info(f"SNLI binary: {len(texts_a)} samples | consistent={labels.count(1)}, contradict={labels.count(0)}")
    return texts_a, texts_b, labels


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, loader, optimizer, scheduler, device, class_weights=None):
    model.train()
    total_loss, steps = 0.0, 0

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels_batch   = batch["labels"].to(device)

        optimizer.zero_grad()

        out  = model(input_ids, attention_mask, token_type_ids, labels_batch, class_weights)
        loss = out["loss"]

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        steps += 1

    return total_loss / steps


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels_batch   = batch["labels"]

        out    = model(input_ids, attention_mask, token_type_ids)
        preds  = torch.argmax(out["logits"], dim=-1).cpu()

        all_preds.extend(preds.tolist())
        all_labels.extend(labels_batch.tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, f1, all_preds, all_labels


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — SNLI pre-training
# ─────────────────────────────────────────────────────────────────────────────

def stage1_snli(
    model,
    tokenizer,
    device,
    max_samples=50000,
    batch_size=32,
    epochs=2,
    lr=3e-4,
    max_len=256,
    save_dir="model_nli",
):
    logger.info("=" * 60)
    logger.info("STAGE 1 — Training on SNLI (from scratch)")
    logger.info("=" * 60)

    texts_a, texts_b, labels = load_snli_binary(max_samples)

    # 90/10 split
    cut = int(0.9 * len(texts_a))
    train_ds = SNLIDataset(texts_a[:cut], texts_b[:cut], labels[:cut], tokenizer, max_len)
    val_ds   = SNLIDataset(texts_a[cut:], texts_b[cut:], labels[cut:], tokenizer, max_len)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_dl) * epochs
    scheduler   = OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.1)

    best_f1 = 0.0
    for epoch in range(1, epochs + 1):
        t0   = time.time()
        loss = train_one_epoch(model, train_dl, optimizer, scheduler, device)
        acc, f1, _, _ = evaluate(model, val_dl, device)
        elapsed = (time.time() - t0) / 60
        logger.info(f"[SNLI] epoch {epoch}/{epochs} | loss={loss:.4f} | acc={acc:.4f} | f1={f1:.4f} | {elapsed:.1f}min")

        if f1 > best_f1:
            best_f1 = f1
            model.save(save_dir)
            logger.info(f"  checkpoint saved (f1={f1:.4f})")

    logger.info(f"Stage 1 done. Best SNLI val F1: {best_f1:.4f}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Book data fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

def stage2_books(
    model,
    tokenizer,
    train_samples,
    device,
    batch_size=8,
    epochs=8,
    lr=5e-5,
    max_len=256,
    save_dir="model_nli",
    seed=42,
):
    logger.info("=" * 60)
    logger.info("STAGE 2 — Fine-tuning on book data")
    logger.info("=" * 60)

    all_labels = [s.label for s in train_samples]

    # stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(all_labels)), all_labels))

    train_data = [train_samples[i] for i in train_idx]
    val_data   = [train_samples[i] for i in val_idx]

    train_ds = BookNLIDataset(train_data, tokenizer, max_len)
    val_ds   = BookNLIDataset(val_data,   tokenizer, max_len)

    # class weights for imbalance
    counts = np.bincount(all_labels, minlength=2)
    total  = len(train_data)
    w0 = total / (2 * counts[0]) if counts[0] > 0 else 1.0
    w1 = total / (2 * counts[1]) if counts[1] > 0 else 1.0
    class_weights = torch.tensor([w0, w1], dtype=torch.float)
    logger.info(f"class weights — contradict: {w0:.3f}, consistent: {w1:.3f}")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info(f"train: {len(train_ds)} | val: {len(val_ds)}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_dl) * epochs
    scheduler   = OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.1)

    best_f1, best_acc = 0.0, 0.0
    id2label = {0: "contradict", 1: "consistent"}

    for epoch in range(1, epochs + 1):
        t0   = time.time()
        loss = train_one_epoch(model, train_dl, optimizer, scheduler, device, class_weights)
        acc, f1, preds, true = evaluate(model, val_dl, device)
        elapsed = (time.time() - t0) / 60
        logger.info(f"[Books] epoch {epoch}/{epochs} | loss={loss:.4f} | acc={acc:.4f} | f1={f1:.4f} | {elapsed:.1f}min")

        if f1 > best_f1:
            best_f1  = f1
            best_acc = acc
            model.save(save_dir)
            logger.info(f"  checkpoint saved (f1={f1:.4f})")

    # print final classification report on val
    _, _, preds, true = evaluate(model, val_dl, device)
    names  = [id2label[i] for i in sorted(id2label.keys())]
    logger.info(f"\n{classification_report(true, preds, target_names=names, zero_division=0)}")
    logger.info(f"Stage 2 done. Best book val F1: {best_f1:.4f}  Acc: {best_acc:.4f}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def train(
    train_samples,
    save_dir=Path("model_nli"),
    tokenizer_name="bert-base-uncased",
    snli_samples=50000,
    snli_epochs=2,
    book_epochs=8,
    batch_size=16,
    max_len=256,
    seed=42,
    skip_snli=False,
):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    torch.manual_seed(seed)
    np.random.seed(seed)

    import os
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    device = get_device()
    logger.info(f"device: {device}")

    # tokenizer only — we are NOT loading any model weights from here
    logger.info(f"loading tokenizer: {tokenizer_name}  (weights not loaded, only vocabulary)")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # build our model entirely from scratch
    logger.info("building NarrativeNLIModel from scratch...")
    model = NarrativeNLIModel(
        vocab_size  = tokenizer.vocab_size,
        embed_dim   = 256,
        num_heads   = 4,
        num_layers  = 4,
        ffn_dim     = 512,
        max_len     = max_len,
        num_labels  = 2,
        dropout     = 0.1,
    )
    model = model.to(device)

    # STAGE 1 — SNLI (general NLI knowledge)
    if not skip_snli:
        model = stage1_snli(
            model, tokenizer, device,
            max_samples=snli_samples,
            batch_size=batch_size,
            epochs=snli_epochs,
            max_len=max_len,
            save_dir=str(save_dir),
        )
    else:
        logger.info("skipping SNLI stage (skip_snli=True)")

    # STAGE 2 — our book data (narrative NLI)
    model = stage2_books(
        model, tokenizer, train_samples, device,
        batch_size=batch_size,
        epochs=book_epochs,
        max_len=max_len,
        save_dir=str(save_dir),
        seed=seed,
    )

    return model, tokenizer


if __name__ == "__main__":
    # quick sanity check — just build the model and print param count
    model = NarrativeNLIModel()
    total, trainable = model.count_parameters()
    print(f"Model ready. Params: {total:,} total / {trainable:,} trainable")
    print("All weights initialised from scratch — no pretrained weights loaded.")
