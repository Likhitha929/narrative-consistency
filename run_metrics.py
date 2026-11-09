"""
run_metrics.py
==============
Comprehensive metrics evaluation for the NarrativeNLIModel.
Uses the ALREADY SAVED model — no retraining needed.

Metrics computed:
  ── Classification ──────────────────────────────────────────
  1.  Accuracy & Balanced Accuracy
  2.  Macro / Weighted F1
  3.  Per-class Precision, Recall, F1
  4.  Matthews Correlation Coefficient (MCC)
  5.  Cohen's Kappa
  6.  ROC-AUC (binary)
  7.  PR-AUC (Average Precision)
  8.  Brier Score (calibration)
  9.  Expected Calibration Error (ECE)
  10. False Positive Rate / False Negative Rate
  11. Error Analysis by Book

  ── NLI / Consistency Checker Specific ──────────────────────
  12. Contradiction Detection Rate
  13. False Alarm Rate (consistent wrongly flagged as contradict)
  14. Miss Rate (contradictions that slipped through)
  15. Confidence gap: correct vs wrong predictions

  ── RAG Retrieval ────────────────────────────────────────────
  16. Semantic Similarity (mean, min, max, std, distribution)
  17. % of pairs with sim > 0.5 (good retrieval)
  18. % of pairs with sim < 0.3 (poor retrieval)

Run:
    python3 run_metrics.py
"""

import json
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    cohen_kappa_score,
    brier_score_loss,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import config
import data_loader
import chunking
import embedding
import retrieval
import dataset_builder
from custom_nli_model import NarrativeNLIModel
from train_scratch import BookDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_metrics")

SEP  = "=" * 60
SEP2 = "-" * 60


# ─── helpers ─────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():        return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def get_predictions_and_probs(model, loader, device):
    """Returns (preds list, trues list, probs ndarray shape (N,2))."""
    model.eval()
    all_preds, all_trues, all_probs = [], [], []
    for batch in loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        ttid = batch["token_type_ids"].to(device)
        out  = model(ids, mask, ttid)
        prob = torch.softmax(out["logits"], dim=-1).cpu().numpy()
        pred = np.argmax(prob, axis=1).tolist()
        all_preds.extend(pred)
        all_trues.extend(batch["labels"].tolist())
        all_probs.append(prob)
    return all_preds, all_trues, np.vstack(all_probs)


def expected_calibration_error(y_true, probs_pos, n_bins=10):
    """ECE — how well predicted probabilities match actual accuracy."""
    bins    = np.linspace(0, 1, n_bins + 1)
    ece     = 0.0
    n       = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs_pos >= lo) & (probs_pos < hi)
        if mask.sum() == 0:
            continue
        bin_acc  = np.mean(np.array(y_true)[mask] == 1)
        bin_conf = np.mean(probs_pos[mask])
        ece     += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return round(float(ece), 4)


def print_section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def pct(val): return f"{val*100:.2f}%"


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{SEP}")
    print("  NarrativeNLI — Comprehensive Metrics Report")
    print(f"{SEP}\n")

    # ── 1. pipeline setup ────────────────────────────────────────────────────
    logger.info("loading books and building retriever...")
    books    = data_loader.load_books(config.BOOKS_DIR)
    passages = chunking.chunk_all_books(books, config.CHUNK_SIZE, config.CHUNK_OVERLAP)

    index_dir = Path("cache/embedding_index")
    if (index_dir / "embeddings.npy").exists():
        emb_index = embedding.load_index(index_dir)
    else:
        emb_index = embedding.build_embedding_index(
            passages, config.EMBEDDING_MODEL_NAME, batch_size=64
        )
        embedding.save_index(emb_index, index_dir)

    retriever   = retrieval.Retriever(emb_index)
    train_df    = data_loader.load_train_csv(config.TRAIN_CSV)
    all_samples = dataset_builder.build_samples(
        train_df, retriever, top_k=config.TOP_K,
        label2id=config.LABEL2ID, is_test=False,
    )

    # ── 2. reproducible val split (same seed as training) ────────────────────
    all_labels = [s.label for s in all_samples]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=config.SEED)
    _, val_idx = next(sss.split(np.zeros(len(all_labels)), all_labels))
    val_samples = [all_samples[i] for i in val_idx]

    logger.info(f"val samples: {len(val_samples)}")

    # ── 3. load saved model ───────────────────────────────────────────────────
    model_dir = config.MODEL_SAVE_DIR
    if not model_dir.exists():
        logger.error(f"No saved model found at '{model_dir}'. Run main.py first.")
        sys.exit(1)

    device    = get_device()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model     = NarrativeNLIModel.load(str(model_dir)).to(device)
    model.eval()
    logger.info(f"loaded model from {model_dir} | device: {device}")

    val_ds = BookDataset(val_samples, tokenizer, config.MAX_SEQ_LEN)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    # ── 4. get predictions ────────────────────────────────────────────────────
    logger.info("running inference on val set...")
    preds, trues, probs = get_predictions_and_probs(model, val_dl, device)

    preds  = np.array(preds)
    trues  = np.array(trues)
    conf   = np.max(probs, axis=1)         # prediction confidence
    prob_1 = probs[:, 1]                   # P(consistent)

    # ── 5. confusion matrix raw ───────────────────────────────────────────────
    cm = confusion_matrix(trues, preds)
    # cm[actual][predicted]:  [0,0]=TN(c,c)  [0,1]=FP  [1,0]=FN  [1,1]=TP
    # label 0=contradict, 1=consistent
    TN = cm[0, 0]   # true contradict
    FP = cm[0, 1]   # contradict predicted as consistent
    FN = cm[1, 0]   # consistent predicted as contradict
    TP = cm[1, 1]   # true consistent

    n_contradict = int((trues == 0).sum())
    n_consistent = int((trues == 1).sum())

    # ═══════════════════════════════════════════════════════
    # SECTION 1: Classification Metrics
    # ═══════════════════════════════════════════════════════
    print_section("1. CLASSIFICATION METRICS")

    bal_acc  = balanced_accuracy_score(trues, preds)
    f1_mac   = f1_score(trues, preds, average="macro",    zero_division=0)
    f1_wt    = f1_score(trues, preds, average="weighted", zero_division=0)
    prec_mac = precision_score(trues, preds, average="macro",    zero_division=0)
    rec_mac  = recall_score(trues, preds, average="macro",    zero_division=0)

    print(f"  Balanced Accuracy     : {pct(bal_acc)}  (accounts for class imbalance)")
    print(f"  Macro F1              : {f1_mac:.4f}")
    print(f"  Weighted F1           : {f1_wt:.4f}")
    print(f"  Macro Precision       : {prec_mac:.4f}")
    print(f"  Macro Recall          : {rec_mac:.4f}")

    # ═══════════════════════════════════════════════════════
    # SECTION 2: Per-Class Metrics
    # ═══════════════════════════════════════════════════════
    print_section("2. PER-CLASS METRICS")
    print(classification_report(
        trues, preds,
        target_names=["contradict", "consistent"],
        zero_division=0,
        digits=4,
    ))

    # ═══════════════════════════════════════════════════════
    # SECTION 3: Advanced Classification Metrics
    # ═══════════════════════════════════════════════════════
    print_section("3. ADVANCED CLASSIFICATION METRICS")

    mcc   = matthews_corrcoef(trues, preds)
    kappa = cohen_kappa_score(trues, preds)
    roc   = roc_auc_score(trues, prob_1)
    ap    = average_precision_score(trues, prob_1)
    brier = brier_score_loss(trues, prob_1)
    ece   = expected_calibration_error(trues, prob_1)

    print(f"  Matthews Corr. Coeff  : {mcc:.4f}  (range -1→1, >0.7 = strong)")
    print(f"  Cohen's Kappa         : {kappa:.4f}  (range 0→1, >0.8 = near-perfect)")
    print(f"  ROC-AUC               : {roc:.4f}  (>0.95 = excellent)")
    print(f"  PR-AUC (Avg Precision): {ap:.4f}  (better than ROC for imbalanced data)")
    print(f"  Brier Score           : {brier:.4f}  (lower=better; 0=perfect, 0.25=random)")
    print(f"  Exp. Calibration Err  : {ece:.4f}  (lower=better; how trustworthy probs are)")

    # ═══════════════════════════════════════════════════════
    # SECTION 4: Error Rate Metrics
    # ═══════════════════════════════════════════════════════
    print_section("4. ERROR RATE METRICS")

    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0   # contradict wrongly predicted as consistent
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0   # consistent wrongly predicted as contradict
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0   # recall for consistent
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0   # recall for contradict

    print(f"  Confusion Matrix:")
    print(f"                       Predicted")
    print(f"             contradict  consistent")
    print(f"  contradict     {TN:5d}       {FP:5d}    (actual: {n_contradict})")
    print(f"  consistent     {FN:5d}       {TP:5d}    (actual: {n_consistent})")
    print()
    print(f"  True Positive Rate    : {pct(TPR)}  (consistent recall)")
    print(f"  True Negative Rate    : {pct(TNR)}  (contradict recall)")
    print(f"  False Positive Rate   : {pct(FPR)}  (contradict misclassified as consistent)")
    print(f"  False Negative Rate   : {pct(FNR)}  (consistent misclassified as contradict)")

    # ═══════════════════════════════════════════════════════
    # SECTION 5: NLI / Consistency Checker Specific Metrics
    # ═══════════════════════════════════════════════════════
    print_section("5. CONSISTENCY CHECKER SPECIFIC METRICS")

    contradiction_detection_rate = TNR          # % of actual contradictions caught
    false_alarm_rate             = FPR          # % of consistent claims falsely flagged
    miss_rate                    = 1 - TNR      # % of contradictions that slipped through

    print(f"  Contradiction Detection Rate : {pct(contradiction_detection_rate)}")
    print(f"    → How often real contradictions are correctly caught")
    print()
    print(f"  False Alarm Rate             : {pct(false_alarm_rate)}")
    print(f"    → How often a consistent claim is wrongly flagged as contradiction")
    print()
    print(f"  Contradiction Miss Rate      : {pct(miss_rate)}")
    print(f"    → How often a real contradiction slips through undetected")
    print()
    print(f"  Class Balance in Val Set     : {n_contradict} contradict / {n_consistent} consistent")
    print(f"  Predicted as Contradict      : {int((preds==0).sum())} samples")
    print(f"  Predicted as Consistent      : {int((preds==1).sum())} samples")

    # ═══════════════════════════════════════════════════════
    # SECTION 6: Confidence / Calibration Analysis
    # ═══════════════════════════════════════════════════════
    print_section("6. MODEL CONFIDENCE ANALYSIS")

    correct_mask   = preds == trues
    incorrect_mask = ~correct_mask

    conf_correct   = conf[correct_mask]
    conf_incorrect = conf[incorrect_mask]

    print(f"  Correct predictions   ({correct_mask.sum():4d}): "
          f"mean conf = {conf_correct.mean():.3f} | "
          f"min = {conf_correct.min():.3f} | max = {conf_correct.max():.3f}")

    if incorrect_mask.sum() > 0:
        print(f"  Incorrect predictions ({incorrect_mask.sum():4d}): "
              f"mean conf = {conf_incorrect.mean():.3f} | "
              f"min = {conf_incorrect.min():.3f} | max = {conf_incorrect.max():.3f}")
        high_conf_wrong = (conf_incorrect > 0.9).sum()
        print(f"  High-conf wrong (>90%): {high_conf_wrong}  ← dangerous overconfident errors")
    else:
        print("  No incorrect predictions! (perfect val accuracy)")

    # ═══════════════════════════════════════════════════════
    # SECTION 7: Error Analysis by Book
    # ═══════════════════════════════════════════════════════
    print_section("7. ERROR ANALYSIS BY BOOK")

    books_in_val = {}
    for i, s in enumerate(val_samples):
        b = s.book_name
        if b not in books_in_val:
            books_in_val[b] = {"total": 0, "correct": 0, "contradict_total": 0, "contradict_correct": 0}
        books_in_val[b]["total"] += 1
        if preds[i] == trues[i]:
            books_in_val[b]["correct"] += 1
        if trues[i] == 0:
            books_in_val[b]["contradict_total"] += 1
            if preds[i] == 0:
                books_in_val[b]["contradict_correct"] += 1

    for book, stats in books_in_val.items():
        book_acc = stats["correct"] / stats["total"]
        cd_rate  = (stats["contradict_correct"] / stats["contradict_total"]
                    if stats["contradict_total"] > 0 else float("nan"))
        print(f"\n  Book: {book[:50]}")
        print(f"    Overall accuracy   : {pct(book_acc)}  ({stats['correct']}/{stats['total']})")
        print(f"    Contradict recall  : {pct(cd_rate) if not math.isnan(cd_rate) else 'N/A'}  "
              f"({stats['contradict_correct']}/{stats['contradict_total']} contradictions caught)")

    # ═══════════════════════════════════════════════════════
    # SECTION 8: RAG Retrieval Quality
    # ═══════════════════════════════════════════════════════
    print_section("8. RAG RETRIEVAL QUALITY (Semantic Similarity)")

    logger.info("computing semantic similarity on val set...")
    try:
        from sentence_transformers import SentenceTransformer
        st_dev   = "mps" if torch.backends.mps.is_available() else "cpu"
        st_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=st_dev)

        claims    = [s.input_text.split(" [SEP] ")[-1] for s in val_samples]
        passages_ = [s.input_text.split(" [SEP] ")[0]  for s in val_samples]

        emb_c = st_model.encode(claims,     normalize_embeddings=True, show_progress_bar=False)
        emb_p = st_model.encode(passages_,  normalize_embeddings=True, show_progress_bar=False)
        sims  = (emb_c * emb_p).sum(axis=1)

        poor   = (sims < 0.3).sum()
        mod    = ((sims >= 0.3) & (sims < 0.5)).sum()
        good   = (sims >= 0.5).sum()

        print(f"  Mean similarity    : {sims.mean():.4f}")
        print(f"  Std  similarity    : {sims.std():.4f}")
        print(f"  Min  similarity    : {sims.min():.4f}")
        print(f"  Max  similarity    : {sims.max():.4f}")
        print()
        print(f"  Quality breakdown:")
        print(f"    Poor  (< 0.30) : {poor:4d} samples  ({poor/len(sims)*100:.1f}%)  ← bad retrieval")
        print(f"    Moderate (0.30–0.50): {mod:4d} samples  ({mod/len(sims)*100:.1f}%)  ← ok")
        print(f"    Good  (≥ 0.50) : {good:4d} samples  ({good/len(sims)*100:.1f}%)  ← strong match")
        print()

        # sim analysis by correct vs wrong predictions
        sim_correct   = sims[correct_mask]
        sim_incorrect = sims[incorrect_mask] if incorrect_mask.sum() > 0 else np.array([])

        print(f"  Sim on correct predictions  : {sim_correct.mean():.4f}")
        if len(sim_incorrect) > 0:
            print(f"  Sim on incorrect predictions: {sim_incorrect.mean():.4f}")
            print(f"  → Lower similarity on wrong predictions = retrieval is hurting accuracy")

    except Exception as e:
        logger.warning(f"Semantic similarity failed: {e}")

    # ═══════════════════════════════════════════════════════
    # FINAL SCORECARD
    # ═══════════════════════════════════════════════════════
    print_section("FINAL SCORECARD")
    grade = lambda v, t1, t2: "🟢 Excellent" if v >= t1 else ("🟡 Good" if v >= t2 else "🔴 Needs Work")

    #print(f"  Balanced Accuracy     : {pct(bal_acc)}     {grade(bal_acc, 0.95, 0.85)}")
    print(f"  Macro F1              : {f1_mac:.4f}")
    print(f"  MCC                   : {mcc:.4f}")
    print(f"  Cohen Kappa           : {kappa:.4f}")
    print(f"  ROC-AUC               : {roc:.4f}")
    print(f"  PR-AUC                : {ap:.4f}")
    print(f"  Brier Score           : {brier:.4f}   {'🟢 Well calibrated' if brier < 0.05 else ('🟡 OK' if brier < 0.10 else '🔴 Overconfident')}")
    print(f"  ECE                   : {ece:.4f}   {'🟢 Well calibrated' if ece < 0.05 else '🟡 Slightly off'}")
    print(f"  Contradiction Catch   : {pct(contradiction_detection_rate)}  {grade(contradiction_detection_rate, 0.90, 0.80)}")
    print(f"  False Alarm Rate      : {pct(false_alarm_rate)}    {'🟢 Low' if false_alarm_rate < 0.05 else ('🟡 Moderate' if false_alarm_rate < 0.15 else '🔴 High')}")
    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
