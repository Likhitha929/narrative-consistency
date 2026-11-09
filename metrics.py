import logging
from typing import List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# ── Classification metrics ────────────────────────────────────────────────────

def compute_classification_metrics(y_true: List[int], y_pred: List[int], labels=None):
    """
    Compute accuracy, precision, recall, F1 (macro).
    y_true / y_pred are integer label lists (0 = contradict, 1 = consistent).
    """
    if labels is None:
        labels = [0, 1]

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)

    results = {
        "accuracy":  round(acc, 4),
        "precision": round(prec, 4),
        "recall":    round(rec, 4),
        "f1_macro":  round(f1, 4),
    }

    logger.info(f"Classification Metrics: {results}")
    return results


def print_classification_report(y_true, y_pred, id2label=None):
    if id2label is None:
        id2label = {0: "contradict", 1: "consistent"}
    names = [id2label[i] for i in sorted(id2label.keys())]
    report = classification_report(y_true, y_pred, target_names=names, zero_division=0)
    print("\n--- Classification Report ---")
    print(report)
    return report


# ── Recall@K ──────────────────────────────────────────────────────────────────

def recall_at_k(relevant_ids: List[set], retrieved_ids: List[List], k: int = 3):
    """
    Recall@K: for each query, checks if at least one relevant passage
    appears in the top-k retrieved results.

    Args:
        relevant_ids  : list of sets — each set contains the ground-truth
                        passage chunk_ids for that query.
        retrieved_ids : list of lists — each list contains chunk_ids of
                        the retrieved passages (in ranked order).
        k             : cutoff

    Returns:
        float — mean Recall@K across all queries
    """
    hits = 0
    for rel_set, ret_list in zip(relevant_ids, retrieved_ids):
        top_k = set(ret_list[:k])
        if top_k & rel_set:   # intersection — at least one hit
            hits += 1

    score = hits / len(relevant_ids) if relevant_ids else 0.0
    logger.info(f"Recall@{k}: {score:.4f}  ({hits}/{len(relevant_ids)} queries hit)")
    return round(score, 4)


# ── Semantic similarity ───────────────────────────────────────────────────────

def compute_semantic_similarity(
    texts_a: List[str],
    texts_b: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    """
    Computes pairwise cosine similarity between two lists of texts.
    Returns the mean similarity and per-pair scores.

    Useful for checking how semantically close retrieved passages are
    to the original claims.

    Args:
        texts_a    : list of strings (e.g. retrieved passages)
        texts_b    : list of strings (e.g. claims)
        model_name : sentence-transformers model to use

    Returns:
        dict with 'mean_similarity' and 'per_pair_scores'
    """
    import torch
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model = SentenceTransformer(model_name, device=device)

    emb_a = model.encode(texts_a, convert_to_numpy=True, normalize_embeddings=True)
    emb_b = model.encode(texts_b, convert_to_numpy=True, normalize_embeddings=True)

    # cosine sim = dot product since both are L2-normalised
    scores = (emb_a * emb_b).sum(axis=1)   # element-wise dot, shape (N,)

    mean_sim = float(scores.mean())
    logger.info(f"Semantic Similarity — mean: {mean_sim:.4f}  min: {scores.min():.4f}  max: {scores.max():.4f}")

    return {
        "mean_similarity": round(mean_sim, 4),
        "per_pair_scores": scores.tolist(),
    }


# ── Convenience: run everything at once ──────────────────────────────────────

def evaluate_all(
    y_true,
    y_pred,
    claims=None,
    retrieved_passage_texts=None,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    k=3,
):
    """
    Run all metrics in one shot and return a combined results dict.

    Args:
        y_true                   : ground-truth labels (int list)
        y_pred                   : predicted labels (int list)
        claims                   : list of claim strings (for semantic sim)
        retrieved_passage_texts  : list of best retrieved passage strings
                                   (one per claim, for semantic sim)
        model_name               : model for semantic similarity
        k                        : k for Recall@K (only if you pass chunk ids separately)

    Returns:
        dict with all computed metrics
    """
    results = {}

    # classification
    clf = compute_classification_metrics(y_true, y_pred)
    results.update(clf)

    # semantic similarity — only if inputs are given
    if claims and retrieved_passage_texts:
        assert len(claims) == len(retrieved_passage_texts), \
            "claims and passages must have same length"
        sim = compute_semantic_similarity(retrieved_passage_texts, claims, model_name)
        results["mean_semantic_similarity"] = sim["mean_similarity"]
        results["per_pair_similarities"]    = sim["per_pair_scores"]

    print("\n=== Evaluation Summary ===")
    for k_name, v in results.items():
        if k_name != "per_pair_similarities":
            print(f"  {k_name}: {v}")
    print("==========================\n")

    return results
