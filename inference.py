import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from custom_nli_model import NarrativeNLIModel

logger = logging.getLogger(__name__)


def _get_device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def find_optimal_threshold(val_probs, val_true, metric="contradiction_recall", verbose=True):
    """
    Sweep decision thresholds on the val set to find the best one.

    metric choices:
      "f1_macro"             → maximise macro-F1
      "contradiction_recall" → maximise recall for class-0 (catch all contradictions)
      "balanced"             → balanced accuracy (good general choice)

    Returns the optimal threshold (float between 0 and 1).
    Threshold is applied to P(consistent): predict consistent if P(consistent) > threshold,
    otherwise predict contradict.
    """
    from sklearn.metrics import f1_score, recall_score, balanced_accuracy_score

    thresholds  = np.arange(0.30, 0.80, 0.01)
    best_thresh = 0.50
    best_score  = -1.0

    trues = np.array(val_true)
    p1    = val_probs[:, 1]          # P(consistent)

    for t in thresholds:
        preds = (p1 >= t).astype(int)   # 1=consistent if P(consistent)>=t, else 0=contradict

        if metric == "f1_macro":
            score = f1_score(trues, preds, average="macro", zero_division=0)
        elif metric == "contradiction_recall":
            # Blend Macro-F1 and Contradiction Recall to naturally favour thresholds
            # that catch contradictions, without setting an impossible hard cutoff.
            f1  = f1_score(trues, preds, average="macro", zero_division=0)
            rec = recall_score(trues, preds, pos_label=0, zero_division=0)
            score = (f1 + rec) / 2.0
        elif metric == "balanced":
            score = balanced_accuracy_score(trues, preds)
        else:
            score = f1_score(trues, preds, average="macro", zero_division=0)

        if score > best_score:
            best_score  = score
            best_thresh = t

    if verbose:
        preds_opt = (p1 >= best_thresh).astype(int)
        from sklearn.metrics import recall_score, f1_score, confusion_matrix
        cm = confusion_matrix(trues, preds_opt)
        rec_c = recall_score(trues, preds_opt, pos_label=0, zero_division=0)
        f1_m  = f1_score(trues, preds_opt, average="macro", zero_division=0)
        logger.info(
            f"optimal threshold: {best_thresh:.2f} | "
            f"contradict_recall={rec_c:.4f} | macro_f1={f1_m:.4f} | "
            f"TP={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TN={cm[1,1]}"
        )

    return float(best_thresh)


def predict(test_samples, model_dir: Path, batch_size=16, max_seq_len=256,
            id2label=None, threshold=0.50):
    """
    Run inference on test_samples.

    threshold : decision boundary for P(consistent).
        < 0.50 → model is MORE sensitive to contradictions (recommended for crime checker).
        Set automatically by find_optimal_threshold() when called from main.py.
    """
    if id2label is None:
        id2label = {0: "contradict", 1: "consistent"}

    if not model_dir.exists():
        raise FileNotFoundError(f"model not found at '{model_dir}' — run training first")

    device    = _get_device()
    logger.info(f"inference on: {device}  |  threshold={threshold:.2f}")

    from train_scratch import get_tokenizer
    tokenizer = get_tokenizer(model_dir)          # the vocabulary the model was trained with
    model     = NarrativeNLIModel.load(str(model_dir)).to(device)
    model.eval()

    all_probs = []
    with torch.no_grad():
        batch_size = min(batch_size, len(test_samples))
        for i in range(0, len(test_samples), batch_size):
            chunk = test_samples[i : i + batch_size]

            texts_a, texts_b = [], []
            for s in chunk:
                parts = s.input_text.rsplit(" [SEP] ", 1)
                texts_a.append(parts[0] if len(parts) == 2 else s.input_text)
                texts_b.append(parts[1] if len(parts) == 2 else "")

            enc = tokenizer(
                texts_a, texts_b,
                max_length=max_seq_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            ids  = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            ttid = enc.get("token_type_ids", torch.zeros_like(ids)).to(device)

            out = model(ids, mask, ttid)
            prob = torch.softmax(out["logits"], dim=-1).cpu().numpy()
            all_probs.append(prob)

    all_probs = np.vstack(all_probs)           # (N, 2)
    preds_int = (all_probs[:, 1] >= threshold).astype(int)   # 1=consistent, 0=contradict
    labels    = [id2label.get(p, "unknown") for p in preds_int]
    logger.info(f"predictions: { {l: labels.count(l) for l in set(labels)} }")
    return labels


def save_predictions(test_df, label_preds, output_path: Path):
    assert len(test_df) == len(label_preds), "mismatch between test rows and predictions"
    out = pd.DataFrame({"id": test_df["id"].values, "label": label_preds})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    logger.info(f"predictions saved → {output_path} ({len(out)} rows)")
