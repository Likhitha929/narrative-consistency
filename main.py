"""
main.py  —  Crime / Narrative NLI Pipeline (Full End-to-End)
=============================================================
Steps:
  1.  Load books
  2.  Chunk into passages
  3.  Build / load embedding index
  4.  Set up RAG retriever
  5.  Build training samples
  6.  Train from-scratch NarrativeNLIModel
  7.  Build test samples & run inference
  8.  Save predictions.csv
  9.  ── Comprehensive Metrics ──────────────────────────────
       • Accuracy, Balanced Accuracy
       • Macro / Weighted F1
       • Per-class Precision / Recall / F1
       • Matthews Correlation Coefficient (MCC)
       • Cohen's Kappa
       • ROC-AUC  &  PR-AUC
       • Brier Score  &  ECE
       • False Alarm Rate / Miss Rate (NLI-specific)
       • Error analysis by book
       • RAG Semantic Similarity quality
  10. ── Evaluation Plots (saved to eval_plots/) ─────────────
       • Train vs Val Loss  (overfitting diagnostic)
       • Generalisation Gap per epoch
       • LR schedule
       • Confusion Matrix
       • Per-class bar chart
       • ROC-AUC curve
       • Precision-Recall curve
       • Calibration curve
       • Confidence histogram
       • Summary scorecard
       • Semantic Similarity distribution

Usage:
    python3 main.py                   # full run (train + eval + plots)
    python3 main.py --skip-training   # reuse saved model
    python3 main.py --rebuild-index   # re-embed passages
"""

import argparse
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

import config
import data_loader
import chunking
import embedding
import retrieval
import dataset_builder
import train_scratch
import inference
import metrics as metrics_module
from splits import claim_groups, grouped_split

# ── plotting imports (non-interactive backend) ───────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc,
    average_precision_score, precision_recall_curve,
    matthews_corrcoef, cohen_kappa_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

# ── style ────────────────────────────────────────────────────────────────────
DARK_BG  = "#0e1117"; CARD_BG = "#1a1d27"; TEXT  = "#e8eaf6"
GRID_COL = "#2a2d3a" ; BLUE   = "#1f77b4" ; ORNG = "#ff7f0e"
GREEN    = "#4caf50" ; RED    = "#f44336" ; PRP  = "#ab47bc"

plt.rcParams.update({
    "figure.facecolor": DARK_BG, "axes.facecolor": CARD_BG,
    "axes.edgecolor": GRID_COL,  "axes.labelcolor": TEXT,
    "axes.titlecolor": TEXT,     "xtick.color": TEXT,
    "ytick.color": TEXT,         "text.color": TEXT,
    "grid.color": GRID_COL,      "grid.linestyle": "--",
    "grid.alpha": 0.4,           "legend.facecolor": CARD_BG,
    "legend.edgecolor": GRID_COL,"font.size": 11,
})

PLOT_DIR = Path("eval_plots")
PLOT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", mode="w"),
    ],
)
logger = logging.getLogger("main")


# ═══════════════════════════════════════════════════════════════════════════════
# Metric helpers
# ═══════════════════════════════════════════════════════════════════════════════

def expected_calibration_error(y_true, probs_pos, n_bins=10):
    bins, ece, n = np.linspace(0, 1, n_bins + 1), 0.0, len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs_pos >= lo) & (probs_pos < hi)
        if mask.sum() == 0: continue
        ece += (mask.sum() / n) * abs(np.mean(np.array(y_true)[mask] == 1) - np.mean(probs_pos[mask]))
    return round(float(ece), 4)


def grade(v, t1=0.95, t2=0.85):
    return "Excellent" if v >= t1 else ("Good" if v >= t2 else "Needs Work")


def pct(v): return f"{v*100:.2f}%"


def save_fig(fig, name):
    p = PLOT_DIR / name
    fig.savefig(p, bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
    logger.info(f"  plot saved → {p}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics  (Section 9)
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_metrics(history, val_samples):
    preds  = np.array(history["val_preds"])
    trues  = np.array(history["val_true"])
    probs  = history["val_probs"]          # (N, 2)
    prob_1 = probs[:, 1]
    conf   = np.max(probs, axis=1)

    cm = confusion_matrix(trues, preds)
    TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    n_contra, n_cons  = int((trues==0).sum()), int((trues==1).sum())

    SEP = "=" * 62

    # ── 1. Core classification ──────────────────────────────────────────────
    print(f"\n{SEP}\n  METRICS REPORT  —  NarrativeNLI Consistency Checker\n{SEP}")

    bal_acc = balanced_accuracy_score(trues, preds)
    f1_mac  = f1_score(trues, preds, average="macro",    zero_division=0)
    f1_wt   = f1_score(trues, preds, average="weighted", zero_division=0)

    print(f"\n── 1. CLASSIFICATION METRICS {'─'*33}")
    print(f"  Balanced Accuracy : {pct(bal_acc)}")
    print(f"  Macro F1          : {f1_mac:.4f}")
    print(f"  Weighted F1       : {f1_wt:.4f}")

    # ── 2. Per-class report ─────────────────────────────────────────────────
    print(f"\n── 2. PER-CLASS REPORT {'─'*40}")
    print(classification_report(trues, preds,
          target_names=["contradict","consistent"], zero_division=0, digits=4))

    # ── 3. Advanced metrics ─────────────────────────────────────────────────
    mcc   = matthews_corrcoef(trues, preds)
    kappa = cohen_kappa_score(trues, preds)
    roc   = roc_auc_score(trues, prob_1)
    ap    = average_precision_score(trues, prob_1)
    brier = brier_score_loss(trues, prob_1)
    ece   = expected_calibration_error(trues, prob_1)

    print(f"── 3. ADVANCED METRICS {'─'*40}")
    print(f"  MCC               : {mcc:.4f}   (>0.7 = strong)")
    print(f"  Cohen's Kappa     : {kappa:.4f}   (>0.8 = near-perfect)")
    print(f"  ROC-AUC           : {roc:.4f}   (>0.95 = excellent)")
    print(f"  PR-AUC            : {ap:.4f}   (better for imbalance)")
    print(f"  Brier Score       : {brier:.4f}   (lower=better)")
    print(f"  Calibration (ECE) : {ece:.4f}   (lower=better)")

    # ── 4. Error rates ──────────────────────────────────────────────────────
    FPR = FP/(FP+TN) if (FP+TN)>0 else 0
    FNR = FN/(FN+TP) if (FN+TP)>0 else 0
    TNR = TN/(TN+FP) if (TN+FP)>0 else 0

    print(f"\n── 4. ERROR RATE METRICS {'─'*38}")
    print(f"  Confusion Matrix:")
    print(f"               Pred:contradict  Pred:consistent")
    print(f"  True:contradict     {TN:5d}           {FP:5d}     (n={n_contra})")
    print(f"  True:consistent     {FN:5d}           {TP:5d}     (n={n_cons})")
    print(f"  FPR (false alarm)  : {pct(FPR)}")
    print(f"  FNR (miss rate)    : {pct(FNR)}")
    print(f"  TNR (contra recall): {pct(TNR)}")

    # ── 5. NLI / consistency-checker specific ───────────────────────────────
    print(f"\n── 5. CONSISTENCY CHECKER METRICS {'─'*29}")
    print(f"  Contradiction Detection Rate : {pct(TNR)}")
    print(f"    → Real contradictions caught correctly")
    print(f"  False Alarm Rate             : {pct(FPR)}")
    print(f"    → Consistent claims wrongly flagged")
    print(f"  Contradiction Miss Rate      : {pct(FNR)}")
    print(f"    → Contradictions that slipped through undetected")

    # ── 6. Confidence analysis ──────────────────────────────────────────────
    correct_mask = preds == trues
    print(f"\n── 6. CONFIDENCE ANALYSIS {'─'*37}")
    print(f"  Correct   ({correct_mask.sum():4d}): mean={conf[correct_mask].mean():.3f}")
    if (~correct_mask).sum() > 0:
        ci = conf[~correct_mask]
        print(f"  Incorrect ({(~correct_mask).sum():4d}): mean={ci.mean():.3f}  |  "
              f"high-conf wrong (>90%): {(ci>0.9).sum()}")

    # ── 7. Error by book ────────────────────────────────────────────────────
    print(f"\n── 7. ERROR ANALYSIS BY BOOK {'─'*34}")
    book_stats = {}
    for i, s in enumerate(val_samples):
        b = s.book_name
        if b not in book_stats:
            book_stats[b] = {"total":0,"correct":0,"cd_total":0,"cd_correct":0}
        book_stats[b]["total"] += 1
        if preds[i] == trues[i]: book_stats[b]["correct"] += 1
        if trues[i] == 0:
            book_stats[b]["cd_total"] += 1
            if preds[i] == 0: book_stats[b]["cd_correct"] += 1
    for book, s in book_stats.items():
        bacc = s["correct"]/s["total"]
        cdr  = s["cd_correct"]/s["cd_total"] if s["cd_total"]>0 else float("nan")
        name = book[:45]
        print(f"  {name:<45}  acc={pct(bacc)}  contradict_recall={pct(cdr) if not math.isnan(cdr) else 'N/A'}")

    # ── 8. Overfitting summary ──────────────────────────────────────────────
    tl = history.get("train_loss", []); vl = history.get("val_loss", [])
    print(f"\n── 8. OVERFITTING SUMMARY {'─'*37}")
    if tl and vl:
        best_ep = int(np.argmin(vl)) + 1
        worst_gap = max(v - t for t, v in zip(tl, vl))
        print(f"  Epochs trained      : {len(tl)}")
        print(f"  Best val_loss epoch : {best_ep}")
        print(f"  Worst gap (val-train): {worst_gap:+.4f}  {'⚠️ overfitting detected' if worst_gap>0.08 else 'within safe range'}")
    else:
        print("  (no training history saved — train once to get curves)")

    # ── Final scorecard ─────────────────────────────────────────────────────
    print(f"\n── FINAL SCORECARD {'─'*44}")
    print(f"  Balanced Accuracy  : {pct(bal_acc)}   {grade(bal_acc)}")
    print(f"  Macro F1           : {f1_mac:.4f}   {grade(f1_mac)}")
    print(f"  MCC                : {mcc:.4f}   {grade(mcc, 0.85, 0.70)}")
    print(f"  Cohen Kappa        : {kappa:.4f}   {grade(kappa, 0.85, 0.70)}")
    print(f"  ROC-AUC            : {roc:.4f}   {grade(roc, 0.97, 0.90)}")
    print(f"  PR-AUC             : {ap:.4f}   {grade(ap, 0.97, 0.90)}")
    print(f"  Brier Score        : {brier:.4f}   {'Well calibrated' if brier<0.05 else ('OK' if brier<0.10 else 'Check calibration')}")
    print(f"  ECE                : {ece:.4f}   {'Well calibrated' if ece<0.05 else 'Moderate'}")
    print(f"  Contradiction Catch: {pct(TNR)}   {grade(TNR, 0.90, 0.80)}")
    print(f"  False Alarm Rate   : {pct(FPR)}   {'Low' if FPR<0.05 else ('Moderate' if FPR<0.15 else 'High')}")
    print(f"\n{SEP}\n")

    return dict(preds=preds, trues=trues, probs=probs, prob_1=prob_1, conf=conf,
                mcc=mcc, kappa=kappa, roc=roc, ap=ap, brier=brier, ece=ece,
                bal_acc=bal_acc, f1_mac=f1_mac,
                FPR=FPR, FNR=FNR, TNR=TNR, correct_mask=correct_mask)


# ═══════════════════════════════════════════════════════════════════════════════
# Plots  (Section 10)
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_plots(history, m, test_samples=None):
    preds, trues, probs = m["preds"], m["trues"], m["probs"]
    prob_1, conf        = m["prob_1"], m["conf"]

    logger.info("generating evaluation plots...")

    # ── Plot 1: Loss + Accuracy curves ──────────────────────────────────────
    has_curves = bool(history.get("train_loss"))
    if not has_curves:
        logger.info("  no training history — skipping loss/gap plots")
    eps = range(1, len(history["train_loss"]) + 1)
    if has_curves:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Overfitting Diagnostic — Loss & Accuracy", fontsize=14, y=1.02)

        ax = axes[0]
        ax.plot(eps, history["train_loss"], color=BLUE, lw=2.5, marker="o", ms=5, label="Train Loss")
        ax.plot(eps, history["val_loss"],   color=ORNG, lw=2.5, marker="s", ms=5, label="Val Loss")
        ax.set_title("Train vs Val Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(); ax.grid(True)

        ax = axes[1]
        ax.plot(eps, history["val_acc"], color=GREEN,  lw=2.5, marker="o", ms=5, label="Val Accuracy")
        ax.plot(eps, history["val_f1"],  color=PRP,    lw=2.5, marker="s", ms=5, label="Val Macro-F1")
        ax.axhline(max(history["val_acc"]), color=GREEN, lw=1, ls="--", alpha=0.6,
                   label=f"Best Acc={max(history['val_acc']):.3f}")
        ax.axhline(max(history["val_f1"]),  color=PRP,   lw=1, ls="--", alpha=0.6,
                   label=f"Best F1={max(history['val_f1']):.3f}")
        ax.set_title("Val Accuracy & F1"); ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05); ax.legend(); ax.grid(True)
        fig.tight_layout(); save_fig(fig, "01_loss_accuracy_curves.png")

        # ── Plot 2: Generalisation gap ───────────────────────────────────────────
        gap = [v - t for t, v in zip(history["train_loss"], history["val_loss"])]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(eps, gap, color=[RED if g > 0.05 else GREEN for g in gap], alpha=0.8)
        ax.axhline(0, color=TEXT, lw=1)
        ax.axhline(0.08, color=ORNG, lw=1.5, ls="--", label="Overfitting threshold (0.08)")
        ax.set_title("Generalisation Gap per Epoch (Val − Train Loss)")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss Gap"); ax.legend(); ax.grid(True, axis="y")
        fig.tight_layout(); save_fig(fig, "02_generalisation_gap.png")

    # ── Plot 3: Confusion matrix ─────────────────────────────────────────────
    cm = confusion_matrix(trues, preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=False, xticklabels=["contradict","consistent"],
                yticklabels=["contradict","consistent"],
                cmap="Blues", ax=ax, linewidths=0.5, linecolor=GRID_COL)
    for i in range(2):
        for j in range(2):
            ax.text(j+.5, i+.5, f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)",
                    ha="center", va="center", fontsize=13,
                    color="white" if cm[i,j]>cm.max()/2 else "black", fontweight="bold")
    ax.set_title("Confusion Matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout(); save_fig(fig, "03_confusion_matrix.png")

    # ── Plot 4: Per-class bar chart ──────────────────────────────────────────
    classes = ["contradict","consistent"]
    prec = precision_score(trues, preds, average=None, zero_division=0)
    rec  = recall_score(trues, preds, average=None, zero_division=0)
    f1   = f1_score(trues, preds, average=None, zero_division=0)
    x, w = np.arange(2), 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for bars, vals, col, lab in [
        (ax.bar(x-w, prec, w, color=BLUE,  alpha=0.85), prec, BLUE,  "Precision"),
        (ax.bar(x,   rec,  w, color=ORNG,  alpha=0.85), rec,  ORNG,  "Recall"),
        (ax.bar(x+w, f1,   w, color=GREEN, alpha=0.85), f1,   GREEN, "F1"),
    ]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
        bars.set_label(lab)
    ax.set_xticks(x); ax.set_xticklabels(classes, fontsize=12)
    ax.set_ylim(0, 1.15); ax.set_title("Per-class Precision / Recall / F1")
    ax.legend(); ax.grid(True, axis="y"); fig.tight_layout()
    save_fig(fig, "04_per_class_metrics.png")

    # ── Plot 5: ROC-AUC ──────────────────────────────────────────────────────
    fpr_arr, tpr_arr, _ = roc_curve(trues, prob_1)
    roc_val = auc(fpr_arr, tpr_arr)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr_arr, tpr_arr, color=BLUE, lw=2.5, label=f"ROC (AUC={roc_val:.3f})")
    ax.fill_between(fpr_arr, tpr_arr, alpha=0.12, color=BLUE)
    ax.plot([0,1],[0,1], color=GRID_COL, lw=1.5, ls="--", label="Random")
    ax.set_title("ROC-AUC Curve"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(); ax.grid(True); fig.tight_layout()
    save_fig(fig, "05_roc_auc.png")

    # ── Plot 6: Precision-Recall ─────────────────────────────────────────────
    prc, rcc, _ = precision_recall_curve(trues, prob_1)
    ap_val = average_precision_score(trues, prob_1)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.step(rcc, prc, color=ORNG, lw=2.5, where="post", label=f"AP={ap_val:.3f}")
    ax.fill_between(rcc, prc, step="post", alpha=0.12, color=ORNG)
    ax.axhline(np.mean(trues), color=GRID_COL, ls="--", lw=1.5,
               label=f"Baseline (prevalence={np.mean(trues):.2f})")
    ax.set_title("Precision-Recall Curve"); ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(True); fig.tight_layout()
    save_fig(fig, "06_precision_recall.png")

    # ── Plot 7: Calibration curve ────────────────────────────────────────────
    prob_true, prob_pred = calibration_curve(trues, prob_1, n_bins=10)
    brier_val = brier_score_loss(trues, prob_1)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(prob_pred, prob_true, color=BLUE, lw=2.5, marker="o", ms=7,
            label=f"Model (Brier={brier_val:.3f})")
    ax.plot([0,1],[0,1], color=GRID_COL, ls="--", lw=1.5, label="Perfect calibration")
    ax.set_title("Calibration Curve (Reliability Diagram)")
    ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction of Positives")
    ax.legend(); ax.grid(True); fig.tight_layout()
    save_fig(fig, "07_calibration_curve.png")

    # ── Plot 8: Confidence histogram ─────────────────────────────────────────
    correct_mask = preds == trues
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(conf[correct_mask],   bins=20, color=GREEN, alpha=0.7,
            label=f"Correct ({correct_mask.sum()})", density=True)
    if (~correct_mask).sum() > 0:
        ax.hist(conf[~correct_mask], bins=20, color=RED, alpha=0.7,
                label=f"Incorrect ({(~correct_mask).sum()})", density=True)
    ax.axvline(conf[correct_mask].mean(),  color=GREEN, lw=2, ls="--")
    if (~correct_mask).sum() > 0:
        ax.axvline(conf[~correct_mask].mean(), color=RED,   lw=2, ls="--")
    ax.set_title("Prediction Confidence: Correct vs Incorrect")
    ax.set_xlabel("Max Softmax Probability"); ax.set_ylabel("Density")
    ax.legend(); ax.grid(True, axis="y"); fig.tight_layout()
    save_fig(fig, "08_confidence_histogram.png")

    # ── Plot 9: Summary scorecard ────────────────────────────────────────────
    labels_sc = ["Bal. Accuracy","Macro-F1","ROC-AUC","PR-AUC","MCC","Cohen κ"]
    values_sc = [m["bal_acc"], m["f1_mac"], m["roc"], m["ap"], m["mcc"], m["kappa"]]
    colors_sc = [GREEN if v>=0.90 else (ORNG if v>=0.70 else RED) for v in values_sc]
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.barh(labels_sc, values_sc, color=colors_sc, alpha=0.85, height=0.5)
    ax.axvline(0.90, color=GREEN, lw=1.5, ls="--", alpha=0.6, label="Good threshold (0.90)")
    for bar, v in zip(bars, values_sc):
        ax.text(v+0.005, bar.get_y()+bar.get_height()/2, f"{v:.3f}",
                va="center", fontsize=11, color=TEXT, fontweight="bold")
    ax.set_xlim(min(min(values_sc)-0.1, 0), 1.1)
    ax.set_title("Summary Metrics Scorecard"); ax.legend(); ax.grid(True, axis="x")
    fig.tight_layout(); save_fig(fig, "09_summary_scorecard.png")

    # ── Plot 10: Semantic similarity (test set) ──────────────────────────────
    if test_samples:
        try:
            from sentence_transformers import SentenceTransformer
            dev = "mps" if torch.backends.mps.is_available() else "cpu"
            stm = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=dev)
            claims_ = [s.input_text.split(" [SEP] ")[-1] for s in test_samples]
            passg_  = [s.input_text.split(" [SEP] ")[0]  for s in test_samples]
            ec = stm.encode(claims_, normalize_embeddings=True, show_progress_bar=False)
            ep = stm.encode(passg_,  normalize_embeddings=True, show_progress_bar=False)
            sims = (ec * ep).sum(axis=1)

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle("RAG Semantic Similarity — Retrieved Passage vs Claim", fontsize=14)

            ax = axes[0]
            ax.hist(sims, bins=25, color=BLUE, alpha=0.8, edgecolor=GRID_COL)
            ax.axvline(sims.mean(),   color=ORNG,  lw=2, ls="--", label=f"Mean={sims.mean():.3f}")
            ax.axvline(np.median(sims),color=GREEN, lw=2, ls="--", label=f"Median={np.median(sims):.3f}")
            ax.set_title("Cosine Similarity Distribution"); ax.legend(); ax.grid(True, axis="y")

            ax = axes[1]
            ss = np.sort(sims)
            cc = [RED if s<0.3 else (ORNG if s<0.5 else GREEN) for s in ss]
            ax.scatter(range(len(ss)), ss, c=cc, s=18, alpha=0.7)
            ax.axhline(0.5, color=GREEN, ls="--", lw=1.5, label="Good (≥0.5)")
            ax.axhline(0.3, color=ORNG,  ls="--", lw=1.5, label="Moderate (≥0.3)")
            ax.set_title("Per-sample Similarity (sorted)"); ax.legend(); ax.grid(True)

            fig.tight_layout(); save_fig(fig, "10_semantic_similarity.png")
            logger.info(f"Semantic Similarity — mean={sims.mean():.4f}  "
                        f"min={sims.min():.4f}  max={sims.max():.4f}  "
                        f"good(≥0.5)={(sims>=0.5).sum()}/{len(sims)}")
        except Exception as e:
            logger.warning(f"Semantic similarity plot skipped: {e}")

    logger.info(f"All plots saved to: {PLOT_DIR.resolve()}/")


# ═══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def report_holdout(y_true, probs, threshold, holdout_data, train_data, blank_probs=None):
    """
    Score the held-out test split once — at the default 0.50 threshold and at
    the threshold tuned on val — next to a claim-only baseline trained on the
    same train split. This is the number to quote.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    def scores(pred, p1=None):
        out = {"accuracy": float((pred == y_true).mean()),
               "balanced_acc": float(balanced_accuracy_score(y_true, pred)),
               "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
               "mcc": float(matthews_corrcoef(y_true, pred)),
               "contradiction_recall": float(recall_score(y_true, pred, pos_label=0, zero_division=0))}
        if p1 is not None:
            out["roc_auc"] = float(roc_auc_score(y_true, p1))
        return out

    p1 = probs[:, 1]
    model_050 = scores((p1 >= 0.50).astype(int), p1)
    model_opt = scores((p1 >= threshold).astype(int), p1)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    # the baseline sees exactly the claim text the model sees (incl. character name)
    claim_txt = lambda d: [s.input_text.rsplit(" [SEP] ", 1)[-1] for s in d]
    Xtr = vec.fit_transform(claim_txt(train_data))
    clf = LogisticRegression(max_iter=3000, C=4.0).fit(Xtr, [s.label for s in train_data])
    bp  = clf.predict_proba(vec.transform(claim_txt(holdout_data)))[:, 1]
    base = scores((bp >= 0.5).astype(int), bp)

    SEP = "=" * 62
    print(f"\n{SEP}\n  HELD-OUT TEST  (n={len(y_true)}, no claim overlap with train/val)\n{SEP}")
    print(f"  {'':<34}{'acc':>8}{'macro-F1':>10}{'ROC-AUC':>9}")
    rows = [("claim-only TF-IDF baseline", base),
            ("NLI model @ 0.50", model_050),
            (f"NLI model @ {threshold:.2f} (tuned on val)", model_opt)]
    blank = None
    if blank_probs is not None:
        blank = scores((blank_probs[:, 1] >= 0.50).astype(int), blank_probs[:, 1])
        rows.append(("ablation: passages blanked @ 0.50", blank))
    for name, r in rows:
        print(f"  {name:<34}{r['accuracy']*100:7.2f}%{r['macro_f1']:10.4f}{r.get('roc_auc', float('nan')):9.4f}")
    print(f"{SEP}\n")

    return {"test_threshold_050": model_050,
            "test_threshold_tuned": {**model_opt, "threshold": float(threshold)},
            "test_claim_only_baseline": base,
            **({"test_ablation_no_passages": blank} if blank is not None else {})}


def _evidence_tok():
    """Book tokenizer for sentence-level evidence packing (None -> whole passages)."""
    if not config.SENTENCE_EVIDENCE:
        return None
    from book_tokenizer import load_tokenizer
    return load_tokenizer()


def run(skip_training=False, rebuild_index=False, resume=False):
    start = time.time()

    # ── Step 1-2: books → chunks ─────────────────────────────────────────────
    logger.info("loading books...")
    books = data_loader.load_books(config.BOOKS_DIR)
    logger.info("chunking books...")
    passages = chunking.chunk_all_books(books, config.CHUNK_SIZE, config.CHUNK_OVERLAP)

    # ── Step 3: embedding index ──────────────────────────────────────────────
    index_dir = Path("cache/embedding_index")
    if not rebuild_index and (index_dir / "embeddings.npy").exists():
        logger.info("loading cached embedding index...")
        emb_index = embedding.load_index(index_dir)
    else:
        logger.info("building embedding index...")
        emb_index = embedding.build_embedding_index(
            passages, config.EMBEDDING_MODEL_NAME, batch_size=64)
        embedding.save_index(emb_index, index_dir)

    # ── Step 4: retriever ────────────────────────────────────────────────────
    logger.info("setting up retriever...")
    retriever = retrieval.Retriever(emb_index)

    # ── Step 5: training samples ─────────────────────────────────────────────
    logger.info("building training samples...")
    train_df      = data_loader.load_train_csv(config.TRAIN_CSV)
    train_samples = dataset_builder.build_samples(
        train_df, retriever, top_k=config.TOP_K,
        label2id=config.LABEL2ID, is_test=False, with_char=config.WITH_CHAR,
        evidence_tokenizer=_evidence_tok(), max_len=config.MAX_SEQ_LEN)

    if not train_samples:
        logger.error("no training samples — check your data!")
        sys.exit(1)

    # ── Step 5b: leakage-free split ──────────────────────────────────────────
    # Paraphrased copies of a claim are grouped and kept on one side only.
    labels = np.array([s.label for s in train_samples])
    groups = claim_groups([s.claim for s in train_samples],
                          [s.book_name for s in train_samples])
    tr_idx, va_idx, te_idx = grouped_split(labels, groups, config.VAL_FRAC,
                                           config.TEST_FRAC, config.SEED)
    train_data   = [train_samples[i] for i in tr_idx]
    val_data     = [train_samples[i] for i in va_idx]
    holdout_data = [train_samples[i] for i in te_idx]
    logger.info(f"grouped split: {len(set(groups))} distinct claims | "
                f"train={len(train_data)} val={len(val_data)} held-out test={len(holdout_data)}")

    # ── Step 5c: self-supervised pairs mined from the books (train only) ─────
    train_groups = groups[tr_idx]
    real_train   = train_data          # baseline + reporting use real claims only
    if config.USE_SYNTH:
        import pickle, synth_data
        synth_path = Path(config.SYNTH_PATH)
        if synth_path.exists():
            synth = pickle.load(open(synth_path, "rb"))
        else:
            logger.info("building synthetic name/number/negation pairs from the books...")
            synth = synth_data.build(config.SYNTH_N)
            pickle.dump(synth, open(synth_path, "wb"))
        train_data   = train_data + synth
        train_groups = np.concatenate([train_groups, np.arange(len(synth)) + groups.max() + 1])
        logger.info(f"added {len(synth)} synthetic training pairs")

    from train_scratch import (BookDataset, get_device, get_tokenizer,
                               predict_with_probs, NarrativeNLIModel)
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    import json

    # ── Step 6: train (or load) ──────────────────────────────────────────────
    if not skip_training:
        if config.PRETRAINED and not (Path(config.PRETRAIN_DIR) / "encoder.pt").exists():
            sys.exit("no pretrained encoder found — run `python book_tokenizer.py && "
                     "python pretrain_mlm.py` first, or set PRETRAINED=False in config.py")
        logger.info("training NarrativeNLIModel...")
        best_model, tokenizer, history, val_dl = train_scratch.train(
            train_data=train_data,
            val_data=val_data,
            save_dir=config.MODEL_SAVE_DIR,
            tokenizer_name=config.TOKENIZER_DIR if config.PRETRAINED else config.TOKENIZER_NAME,
            batch_size=config.BATCH_SIZE,
            epochs=config.NUM_EPOCHS,
            max_len=config.MAX_SEQ_LEN,
            seed=config.SEED,
            patience=config.PATIENCE,
            resume=resume,
            train_groups=train_groups if config.GROUP_SAMPLING else None,
            pretrained_dir=config.PRETRAIN_DIR if config.PRETRAINED else None,
            arch={k: v for k, v in config.MODEL_ARCH.items() if k != "dropout"} if config.PRETRAINED else None,
            **config.FINETUNE,
        )
        device = get_device()
    else:
        logger.info(f"skipping training, using saved model from {config.MODEL_SAVE_DIR}")
        device     = get_device()
        tokenizer  = get_tokenizer(config.MODEL_SAVE_DIR)
        best_model = NarrativeNLIModel.load(str(config.MODEL_SAVE_DIR)).to(device)
        val_dl     = DataLoader(BookDataset(val_data, tokenizer, config.MAX_SEQ_LEN),
                                batch_size=32, shuffle=False, num_workers=0)
        vp, vt, vprob = predict_with_probs(best_model, val_dl, device)
        # curves are only available if a training run saved them
        hist_path = config.MODEL_SAVE_DIR / "history.json"
        curves = json.loads(hist_path.read_text()) if hist_path.exists() else \
                 {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
        history = {**curves, "val_preds": vp, "val_true": vt, "val_probs": vprob}
    history["val_samples"] = val_data

    # ── Step 6b: held-out test probabilities (scored once, never tuned on) ──
    ho_dl = DataLoader(BookDataset(holdout_data, tokenizer, config.MAX_SEQ_LEN),
                       batch_size=32, shuffle=False, num_workers=0)
    _, ho_true, ho_probs = predict_with_probs(best_model, ho_dl, device)
    ho_true = np.array(ho_true)

    # ablation: same model, passages removed -> does it actually read the book?
    from dataclasses import replace
    blank_data = [replace(s, input_text=" [SEP] " + s.input_text.rsplit(" [SEP] ", 1)[-1]) for s in holdout_data]
    blank_dl = DataLoader(BookDataset(blank_data, tokenizer, config.MAX_SEQ_LEN),
                          batch_size=32, shuffle=False, num_workers=0)
    _, _, blank_probs = predict_with_probs(best_model, blank_dl, device)

    # ── Step 7-8: test inference ─────────────────────────────────────────────
    logger.info("building test samples...")
    test_df      = data_loader.load_test_csv(config.TEST_CSV)
    test_samples = dataset_builder.build_samples(
        test_df, retriever, top_k=config.TOP_K,
        label2id=config.LABEL2ID, is_test=True, with_char=config.WITH_CHAR,
        evidence_tokenizer=_evidence_tok(), max_len=config.MAX_SEQ_LEN)

    logger.info("running inference with optimised threshold...")
    preds_out = inference.predict(
        test_samples=test_samples,
        model_dir=config.MODEL_SAVE_DIR,
        batch_size=config.BATCH_SIZE,
        max_seq_len=config.MAX_SEQ_LEN,
        id2label=config.ID2LABEL,
        threshold=0.50,          
    )
    inference.save_predictions(test_df, preds_out, config.PREDICTIONS_PATH)

    # ── Step 9: comprehensive metrics on val set ─────────────────────────────
    logger.info("computing comprehensive metrics...")
    m = run_all_metrics(history, history["val_samples"])

    # ── Step 9b: threshold optimisation (no retraining needed) ───────────────
    logger.info("finding optimal decision threshold on val set...")
    opt_threshold = inference.find_optimal_threshold(
        val_probs = history["val_probs"],
        val_true  = history["val_true"],
        metric    = "contradiction_recall",
        verbose   = True,
    )
    print(f"\n  Optimal threshold  : {opt_threshold:.2f}  (default=0.50)")
    print(f"  Effect: classify as contradict if P(consistent) < {opt_threshold:.2f}")
    print(f"  Re-saving predictions with optimised threshold...\n")

    # re-run inference with optimised threshold and overwrite predictions
    preds_opt = inference.predict(
        test_samples=test_samples,
        model_dir=config.MODEL_SAVE_DIR,
        batch_size=config.BATCH_SIZE,
        max_seq_len=config.MAX_SEQ_LEN,
        id2label=config.ID2LABEL,
        threshold=0.50,          # the val-tuned threshold does not transfer to held-out claims
    )
    inference.save_predictions(test_df, preds_opt, config.PREDICTIONS_PATH)

    # ── Step 9c: held-out test report — the number to quote ─────────────────
    results = report_holdout(ho_true, ho_probs, opt_threshold, holdout_data, real_train, blank_probs)
    results["val"] = {"macro_f1": float(m["f1_mac"]), "balanced_acc": float(m["bal_acc"]),
                      "roc_auc": float(m["roc"]), "n": int(len(val_data))}
    results["split"] = {"train": len(real_train), "synthetic_train": len(train_data) - len(real_train), "val": len(val_data),
                        "test": len(holdout_data), "distinct_claims": int(len(set(groups)))}
    Path("results.json").write_text(json.dumps(results, indent=2))
    logger.info("held-out results -> results.json")

    # ── Step 10: evaluation plots ─────────────────────────────────────────────
    logger.info("generating evaluation plots...")
    run_all_plots(history, m, test_samples=test_samples)

    elapsed = (time.time() - start) / 60
    logger.info(f"\ndone! took {elapsed:.1f} min")
    logger.info(f"predictions  -> {config.PREDICTIONS_PATH}  (threshold=0.50)")
    logger.info(f"plots        -> {PLOT_DIR.resolve()}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-training", action="store_true",
                        help="use existing model, skip training")
    parser.add_argument("--rebuild-index", action="store_true",
                        help="re-build embedding index even if cache exists")
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted training run from model/last.pt")
    args = parser.parse_args()
    run(skip_training=args.skip_training, rebuild_index=args.rebuild_index,
        resume=args.resume)
