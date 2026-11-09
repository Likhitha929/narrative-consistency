"""
evaluate_plots.py
=================
Full diagnostic evaluation suite for the NarrativeNLIModel.

Generates:
  1. Train vs Val Loss  (overfitting check)
  2. Train vs Val Accuracy (overfitting check)
  3. Learning-rate schedule
  4. Confusion Matrix
  5. Per-class Precision / Recall / F1 bar chart
  6. ROC-AUC curve
  7. Precision-Recall curve
  8. Calibration curve (reliability diagram)
  9. Prediction confidence histogram
 10. Matthews Correlation Coefficient (MCC)
 11. Cohen's Kappa
 12. Semantic Similarity distribution (histogram)

Run:
    python evaluate_plots.py

It reuses main.py's pipeline (books → chunks → index → retriever → samples)
so you don't re-build anything if the cache already exists.
"""

import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    matthews_corrcoef,
    cohen_kappa_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use("Agg")          # non-interactive backend, safe on macOS / servers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
from transformers import AutoTokenizer

import config
import data_loader
import chunking
import embedding
import retrieval
import dataset_builder
from custom_nli_model import NarrativeNLIModel
from train_scratch import BookDataset, get_device, run_epoch

# ── output directory ─────────────────────────────────────────────────────────
OUT_DIR = Path("eval_plots")
OUT_DIR.mkdir(exist_ok=True)

# ── style ────────────────────────────────────────────────────────────────────
PALETTE   = "#1f77b4"        # mpl blue
ACCENT    = "#ff7f0e"        # mpl orange
DARK_BG   = "#0e1117"
CARD_BG   = "#1a1d27"
TEXT_COL  = "#e8eaf6"
GRID_COL  = "#2a2d3a"
GREEN     = "#4caf50"
RED       = "#f44336"
PURPLE    = "#ab47bc"

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    CARD_BG,
    "axes.edgecolor":    GRID_COL,
    "axes.labelcolor":   TEXT_COL,
    "axes.titlecolor":   TEXT_COL,
    "xtick.color":       TEXT_COL,
    "ytick.color":       TEXT_COL,
    "text.color":        TEXT_COL,
    "grid.color":        GRID_COL,
    "grid.linestyle":    "--",
    "grid.alpha":        0.4,
    "legend.facecolor":  CARD_BG,
    "legend.edgecolor":  GRID_COL,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("eval_plots")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_fig(fig, name: str):
    path = OUT_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
    logger.info(f"  saved → {path}")
    plt.close(fig)


@torch.no_grad()
def predict_with_probs(model, loader, device):
    """Returns preds, trues, and softmax probs (N, num_classes)."""
    model.eval()
    preds, trues, probs_list = [], [], []
    for batch in loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        ttid = batch["token_type_ids"].to(device)
        out  = model(ids, mask, ttid)
        logits = out["logits"]
        prob   = torch.softmax(logits, dim=-1)
        preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        trues.extend(batch["labels"].tolist())
        probs_list.append(prob.cpu())
    return preds, trues, torch.cat(probs_list, dim=0).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Training with history collection
# ─────────────────────────────────────────────────────────────────────────────

def train_with_history(
    train_samples,
    tokenizer_name="bert-base-uncased",
    batch_size=16,
    epochs=20,
    lr=5e-4,
    max_len=256,
    seed=42,
    accum_steps=2,
    patience=6,
):
    """
    Identical to train_scratch.train() but also returns per-epoch history
    (train_loss, val_loss, val_acc, val_f1, lr_schedule) for plotting.
    """
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]      = "1"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    torch.manual_seed(seed)
    np.random.seed(seed)

    device    = get_device()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # ── stratified 80/20 split ─────────────────────────────────────────────
    all_labels = [s.label for s in train_samples]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(all_labels)), all_labels))

    train_data = [train_samples[i] for i in train_idx]
    val_data   = [train_samples[i] for i in val_idx]

    train_ds = BookDataset(train_data, tokenizer, max_len)
    val_ds   = BookDataset(val_data,   tokenizer, max_len)

    # ── class weights ──────────────────────────────────────────────────────
    counts = np.bincount([s.label for s in train_data], minlength=2)
    total  = len(train_data)
    w0 = total / (2 * counts[0]) if counts[0] > 0 else 1.0
    w1 = total / (2 * counts[1]) if counts[1] > 0 else 1.0
    class_weights = torch.tensor([w0, w1], dtype=torch.float).to(device)

    sample_weights = torch.tensor(
        [w0 if s.label == 0 else w1 for s in train_data], dtype=torch.float
    )
    sampler  = WeightedRandomSampler(sample_weights, num_samples=len(train_data), replacement=True)
    train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # ── model ──────────────────────────────────────────────────────────────
    model = NarrativeNLIModel(
        vocab_size  = tokenizer.vocab_size,
        embed_dim   = 256,
        num_heads   = 8,
        num_layers  = 6,
        ffn_dim     = 1024,
        max_len     = max_len,
        num_labels  = 2,
        dropout     = 0.15,
        focal_gamma = 2.0,
    ).to(device)

    optimizer    = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps  = (len(train_dl) // accum_steps + 1) * epochs
    warmup_steps = max(1, int(0.1 * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # ── collect LR schedule (theoretical) ─────────────────────────────────
    lr_schedule = [lr_lambda(s) * lr for s in range(total_steps)]

    # ── history ────────────────────────────────────────────────────────────
    history = {
        "train_loss": [], "val_loss": [],
        "val_acc":    [], "val_f1":   [],
        "val_preds":  None, "val_true": None,
        "val_probs":  None,
    }

    best_f1    = 0.0
    no_improve = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = run_epoch(model, train_dl, optimizer, scheduler, device,
                               class_weights, training=True, accum_steps=accum_steps)

        # ── validation loss ────────────────────────────────────────────────
        val_loss = run_epoch(model, val_dl, None, None, device,
                             class_weights, training=False, accum_steps=1)

        # ── validation predictions ─────────────────────────────────────────
        val_preds, val_true, val_probs = predict_with_probs(model, val_dl, device)

        acc = accuracy_score(val_true, val_preds)
        f1  = f1_score(val_true, val_preds, average="macro", zero_division=0)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(acc)
        history["val_f1"].append(f1)

        elapsed = (time.time() - t0) / 60
        logger.info(
            f"epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={acc:.4f} | val_f1={f1:.4f} | {elapsed:.1f}min"
        )

        if f1 >= best_f1:
            best_f1    = f1
            no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            history["val_preds"] = val_preds
            history["val_true"]  = val_true
            history["val_probs"] = val_probs
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("  early stopping triggered.")
                break

    # reload best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, tokenizer, history, lr_schedule, val_dl


# ─────────────────────────────────────────────────────────────────────────────
# Plot functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curves(history):
    """
    Plot 1 & 2: Train vs Val Loss  +  Val Accuracy & F1
    Side by side — primary overfitting diagnostic.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Overfitting Diagnostic — Loss & Accuracy Curves", fontsize=15, y=1.02)

    epochs = range(1, len(history["train_loss"]) + 1)

    # ── Loss ──────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], color=PALETTE,  lw=2.5, marker="o", ms=5, label="Train Loss")
    ax.plot(epochs, history["val_loss"],   color=ACCENT,   lw=2.5, marker="s", ms=5, label="Val Loss")

    gap = [v - t for t, v in zip(history["train_loss"], history["val_loss"])]
    worst_gap_idx = int(np.argmax(gap))
    ax.annotate(
        f"Gap={gap[worst_gap_idx]:.3f}",
        xy=(worst_gap_idx + 1, history["val_loss"][worst_gap_idx]),
        xytext=(worst_gap_idx + 1, history["val_loss"][worst_gap_idx] + 0.05),
        arrowprops=dict(arrowstyle="->", color=TEXT_COL),
        fontsize=10, color=TEXT_COL,
    )

    ax.set_title("Train vs Val Loss (Focal Loss)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)

    # ── Accuracy & F1 ─────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(epochs, history["val_acc"], color=GREEN,  lw=2.5, marker="o", ms=5, label="Val Accuracy")
    ax.plot(epochs, history["val_f1"],  color=PURPLE, lw=2.5, marker="s", ms=5, label="Val Macro-F1")
    ax.axhline(max(history["val_acc"]), color=GREEN,  lw=1, ls="--", alpha=0.6,
               label=f"Best Acc={max(history['val_acc']):.3f}")
    ax.axhline(max(history["val_f1"]),  color=PURPLE, lw=1, ls="--", alpha=0.6,
               label=f"Best F1={max(history['val_f1']):.3f}")

    ax.set_title("Val Accuracy & Macro-F1 per Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    save_fig(fig, "01_loss_accuracy_curves.png")


def plot_overfitting_gap(history):
    """
    Plot 3: Generalisation gap  (val_loss − train_loss) over epochs.
    If gap grows large → overfitting.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    gap = [v - t for t, v in zip(history["train_loss"], history["val_loss"])]

    ax.bar(epochs, gap, color=[RED if g > 0.05 else GREEN for g in gap], alpha=0.8, label="Val − Train Loss Gap")
    ax.axhline(0, color=TEXT_COL, lw=1)
    ax.axhline(0.05, color=ACCENT, lw=1.5, ls="--", label="Overfitting threshold (0.05)")

    ax.set_title("Generalisation Gap per Epoch\n(Val Loss − Train Loss)", fontsize=13)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss Gap")
    ax.legend()
    ax.grid(True, axis="y")
    fig.tight_layout()
    save_fig(fig, "02_generalisation_gap.png")


def plot_lr_schedule(lr_schedule):
    """Plot 4: LR warmup + cosine decay."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(lr_schedule, color=PALETTE, lw=2)
    ax.fill_between(range(len(lr_schedule)), lr_schedule, alpha=0.15, color=PALETTE)
    ax.set_title("Learning-Rate Schedule (Warmup + Cosine Decay)", fontsize=13)
    ax.set_xlabel("Optimizer Steps")
    ax.set_ylabel("LR")
    ax.grid(True)
    fig.tight_layout()
    save_fig(fig, "03_lr_schedule.png")


def plot_confusion_matrix(y_true, y_pred):
    """Plot 5: Confusion matrix with counts + percentage."""
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=False, fmt="d",
        xticklabels=["contradict", "consistent"],
        yticklabels=["contradict", "consistent"],
        cmap="Blues", ax=ax, linewidths=0.5, linecolor=GRID_COL,
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j + 0.5, i + 0.5,
                f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                ha="center", va="center",
                fontsize=13, color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontweight="bold",
            )

    ax.set_title("Confusion Matrix", fontsize=14)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    save_fig(fig, "04_confusion_matrix.png")


def plot_per_class_metrics(y_true, y_pred):
    """Plot 6: Per-class Precision / Recall / F1 bar chart."""
    classes = ["contradict", "consistent"]
    prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec  = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1   = f1_score(y_true, y_pred, average=None, zero_division=0)

    x    = np.arange(len(classes))
    w    = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w, prec, w, label="Precision", color=PALETTE,  alpha=0.85)
    b2 = ax.bar(x,     rec,  w, label="Recall",    color=ACCENT,   alpha=0.85)
    b3 = ax.bar(x + w, f1,   w, label="F1",        color=GREEN,    alpha=0.85)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=10, color=TEXT_COL)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title("Per-class Precision / Recall / F1", fontsize=13)
    ax.legend()
    ax.grid(True, axis="y")
    fig.tight_layout()
    save_fig(fig, "05_per_class_metrics.png")


def plot_roc_curve(y_true, probs):
    """Plot 7: ROC-AUC curve for the 'consistent' class."""
    fpr, tpr, _ = roc_curve(y_true, probs[:, 1])
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, color=PALETTE, lw=2.5, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.fill_between(fpr, tpr, alpha=0.12, color=PALETTE)
    ax.plot([0, 1], [0, 1], color=GRID_COL, lw=1.5, ls="--", label="Random classifier")

    ax.set_title("ROC-AUC Curve", fontsize=13)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    save_fig(fig, "06_roc_auc.png")


def plot_pr_curve(y_true, probs):
    """Plot 8: Precision-Recall curve."""
    prec, rec, _ = precision_recall_curve(y_true, probs[:, 1])
    ap = average_precision_score(y_true, probs[:, 1])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.step(rec, prec, color=ACCENT, lw=2.5, where="post",
            label=f"AP = {ap:.3f}")
    ax.fill_between(rec, prec, step="post", alpha=0.12, color=ACCENT)
    baseline = np.mean(y_true)
    ax.axhline(baseline, color=GRID_COL, ls="--", lw=1.5, label=f"Baseline (prevalence={baseline:.2f})")

    ax.set_title("Precision-Recall Curve", fontsize=13)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    save_fig(fig, "07_precision_recall.png")


def plot_calibration(y_true, probs):
    """
    Plot 9: Calibration curve (reliability diagram).
    A well-calibrated model → dots on the diagonal.
    """
    prob_true, prob_pred = calibration_curve(y_true, probs[:, 1], n_bins=10)
    brier = brier_score_loss(y_true, probs[:, 1])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(prob_pred, prob_true, color=PALETTE, lw=2.5, marker="o", ms=7,
            label=f"Model (Brier={brier:.3f})")
    ax.plot([0, 1], [0, 1], color=GRID_COL, ls="--", lw=1.5, label="Perfectly calibrated")

    ax.set_title("Calibration Curve (Reliability Diagram)", fontsize=13)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    save_fig(fig, "08_calibration_curve.png")


def plot_confidence_histogram(y_true, y_pred, probs):
    """
    Plot 10: Confidence histogram — correct vs incorrect predictions.
    Overfitting sign: incorrect predictions have very high confidence too.
    """
    correct   = [i for i, (p, t) in enumerate(zip(y_pred, y_true)) if p == t]
    incorrect = [i for i, (p, t) in enumerate(zip(y_pred, y_true)) if p != t]

    conf = np.max(probs, axis=1)   # max softmax probability

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(conf[correct],   bins=20, color=GREEN, alpha=0.7, label=f"Correct ({len(correct)})",   density=True)
    ax.hist(conf[incorrect], bins=20, color=RED,   alpha=0.7, label=f"Incorrect ({len(incorrect)})", density=True)

    ax.axvline(np.mean(conf[correct]),   color=GREEN, lw=2, ls="--")
    ax.axvline(np.mean(conf[incorrect]), color=RED,   lw=2, ls="--")

    ax.set_title("Prediction Confidence Distribution\n(Correct vs Incorrect)", fontsize=13)
    ax.set_xlabel("Max Softmax Probability (Confidence)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, axis="y")
    fig.tight_layout()
    save_fig(fig, "09_confidence_histogram.png")


def plot_summary_metrics(y_true, y_pred, probs):
    """
    Plot 11: Summary scorecard — MCC, Cohen's κ, ROC-AUC, Brier, AP, F1.
    """
    mcc    = matthews_corrcoef(y_true, y_pred)
    kappa  = cohen_kappa_score(y_true, y_pred)
    roc    = auc(*roc_curve(y_true, probs[:, 1])[:2])
    brier  = brier_score_loss(y_true, probs[:, 1])
    ap     = average_precision_score(y_true, probs[:, 1])
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc    = accuracy_score(y_true, y_pred)

    labels = ["Accuracy", "Macro-F1", "ROC-AUC", "Avg Precision", "MCC", "Cohen's κ"]
    values = [acc, f1_mac, roc, ap, mcc, kappa]
    colors = [GREEN if v >= 0.8 else (ACCENT if v >= 0.6 else RED) for v in values]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.barh(labels, values, color=colors, alpha=0.85, height=0.5)
    ax.axvline(0.8, color=GREEN, lw=1.5, ls="--", alpha=0.6, label="Good threshold (0.8)")
    ax.axvline(0.6, color=ACCENT, lw=1.5, ls="--", alpha=0.6, label="Moderate threshold (0.6)")

    for bar, v in zip(bars, values):
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=11, color=TEXT_COL, fontweight="bold")

    # Brier score annotation (lower is better)
    ax.text(0.02, -0.8, f"Brier Score: {brier:.3f}  (lower = better)", fontsize=10,
            color=TEXT_COL, transform=ax.get_yaxis_transform())

    ax.set_xlim(min(min(values) - 0.1, 0), 1.1)
    ax.set_title("NLI Model — Summary Metrics Scorecard", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, axis="x")
    fig.tight_layout()
    save_fig(fig, "10_summary_scorecard.png")

    logger.info("\n=== METRICS SUMMARY ===")
    logger.info(f"  Accuracy       : {acc:.4f}")
    logger.info(f"  Macro-F1       : {f1_mac:.4f}")
    logger.info(f"  ROC-AUC        : {roc:.4f}")
    logger.info(f"  Avg Precision  : {ap:.4f}")
    logger.info(f"  MCC            : {mcc:.4f}")
    logger.info(f"  Cohen's κ      : {kappa:.4f}")
    logger.info(f"  Brier Score    : {brier:.4f}")
    logger.info("========================\n")


def plot_semantic_similarity(claims, retrieved_passages):
    """
    Plot 12: Distribution of cosine similarities between retrieved
    passages and claims.  Shows whether RAG retrieval is meaningful.
    """
    try:
        from sentence_transformers import SentenceTransformer
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        st_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=device)

        emb_c = st_model.encode(claims,             normalize_embeddings=True)
        emb_p = st_model.encode(retrieved_passages, normalize_embeddings=True)
        scores = (emb_c * emb_p).sum(axis=1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("RAG Semantic Similarity — Retrieved Passage vs Claim", fontsize=14)

        # histogram
        ax = axes[0]
        ax.hist(scores, bins=25, color=PALETTE, alpha=0.8, edgecolor=GRID_COL)
        ax.axvline(np.mean(scores), color=ACCENT, lw=2, ls="--",
                   label=f"Mean = {np.mean(scores):.3f}")
        ax.axvline(np.median(scores), color=GREEN, lw=2, ls="--",
                   label=f"Median = {np.median(scores):.3f}")
        ax.set_title("Cosine Similarity Distribution")
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(True, axis="y")

        # sorted scatter
        ax = axes[1]
        sorted_scores = np.sort(scores)
        c_map = [RED if s < 0.3 else (ACCENT if s < 0.5 else GREEN) for s in sorted_scores]
        ax.scatter(range(len(sorted_scores)), sorted_scores, c=c_map, s=18, alpha=0.7)
        ax.axhline(0.5, color=GREEN,  ls="--", lw=1.5, alpha=0.8, label="Good (≥0.5)")
        ax.axhline(0.3, color=ACCENT, ls="--", lw=1.5, alpha=0.8, label="Moderate (≥0.3)")
        ax.set_title("Per-sample Similarity (sorted)")
        ax.set_xlabel("Sample index (sorted)")
        ax.set_ylabel("Cosine Similarity")
        ax.legend()
        ax.grid(True)

        fig.tight_layout()
        save_fig(fig, "11_semantic_similarity.png")

        logger.info(f"Semantic Similarity — mean={np.mean(scores):.4f} "
                    f"min={np.min(scores):.4f}  max={np.max(scores):.4f}")
    except Exception as e:
        logger.warning(f"Could not compute semantic similarity: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info("=== evaluate_plots.py — NLI Diagnostic Suite ===")

    # ── 1. Pipeline setup (books → chunks → index → retriever → samples) ──
    logger.info("loading books...")
    books = data_loader.load_books(config.BOOKS_DIR)

    logger.info("chunking books...")
    passages = chunking.chunk_all_books(books, config.CHUNK_SIZE, config.CHUNK_OVERLAP)

    index_dir = Path("cache/embedding_index")
    if (index_dir / "embeddings.npy").exists():
        logger.info("loading cached embedding index...")
        emb_index = embedding.load_index(index_dir)
    else:
        logger.info("building embedding index...")
        emb_index = embedding.build_embedding_index(
            passages, config.EMBEDDING_MODEL_NAME, batch_size=64
        )
        embedding.save_index(emb_index, index_dir)

    retriever = retrieval.Retriever(emb_index)

    logger.info("building training samples...")
    train_df = data_loader.load_train_csv(config.TRAIN_CSV)
    train_samples = dataset_builder.build_samples(
        train_df, retriever, top_k=config.TOP_K,
        label2id=config.LABEL2ID, is_test=False,
    )

    if not train_samples:
        logger.error("no training samples — check data!")
        sys.exit(1)

    # ── 2. Train (with history) ────────────────────────────────────────────
    logger.info("starting training with history collection...")
    model, tokenizer, history, lr_schedule, val_dl = train_with_history(
        train_samples,
        batch_size   = config.BATCH_SIZE,
        epochs       = config.NUM_EPOCHS,
        lr           = config.LEARNING_RATE,
        max_len      = config.MAX_SEQ_LEN,
        seed         = config.SEED,
    )

    y_pred = history["val_preds"]
    y_true = history["val_true"]
    probs  = history["val_probs"]   # numpy (N, 2)

    # ── 3. Plots ───────────────────────────────────────────────────────────
    logger.info("\nGenerating plots...")

    plot_loss_curves(history)
    plot_overfitting_gap(history)
    plot_lr_schedule(lr_schedule)
    plot_confusion_matrix(y_true, y_pred)
    plot_per_class_metrics(y_true, y_pred)
    plot_roc_curve(y_true, probs)
    plot_pr_curve(y_true, probs)
    plot_calibration(y_true, probs)
    plot_confidence_histogram(y_true, y_pred, probs)
    plot_summary_metrics(y_true, y_pred, probs)

    # ── 4. Semantic similarity on test set ─────────────────────────────────
    logger.info("computing semantic similarity (test set)...")
    try:
        test_df = data_loader.load_test_csv(config.TEST_CSV)
        test_samples = dataset_builder.build_samples(
            test_df, retriever, top_k=config.TOP_K,
            label2id=config.LABEL2ID, is_test=True,
        )
        top_passages = [s.input_text.split(" [SEP] ")[0]  for s in test_samples]
        claims_text  = [s.input_text.split(" [SEP] ")[-1] for s in test_samples]
        plot_semantic_similarity(claims_text, top_passages)
    except Exception as e:
        logger.warning(f"Semantic similarity plot skipped: {e}")

    # ── 5. Final report ────────────────────────────────────────────────────
    logger.info(f"\n{'='*50}")
    logger.info(f"All plots saved to:  {OUT_DIR.resolve()}/")
    logger.info( "Files:")
    for f in sorted(OUT_DIR.iterdir()):
        logger.info(f"   {f.name}")
    logger.info(f"{'='*50}")

    print(f"\n✅ Done! Open the '{OUT_DIR}/' folder to view all diagnostic plots.")


if __name__ == "__main__":
    main()
