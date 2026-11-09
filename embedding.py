import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingIndex:
    passages:   list
    embeddings: np.ndarray
    model_name: str


def build_embedding_index(passages, model_name, batch_size=64):
    if not passages:
        raise ValueError("no passages to embed!")

    # pick best device available
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    logger.info(f"embedding on: {device}")

    model = SentenceTransformer(model_name, device=device)
    texts = [p.text for p in passages]

    logger.info(f"encoding {len(texts)} passages...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 norm so cosine = dot product later
    )

    return EmbeddingIndex(
        passages=passages,
        embeddings=embeddings.astype(np.float32),
        model_name=model_name,
    )


def save_index(index, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "embeddings.npy", index.embeddings)

    with open(save_dir / "passages.pkl", "wb") as f:
        pickle.dump(index.passages, f)

    meta = {"model_name": index.model_name, "n_passages": len(index.passages)}
    (save_dir / "meta.json").write_text(json.dumps(meta))
    logger.info(f"index saved to {save_dir}")


def load_index(save_dir: Path):
    embeddings = np.load(save_dir / "embeddings.npy")
    with open(save_dir / "passages.pkl", "rb") as f:
        passages = pickle.load(f)
    meta = json.loads((save_dir / "meta.json").read_text())
    logger.info(f"loaded index: {meta['n_passages']} passages")
    return EmbeddingIndex(passages=passages, embeddings=embeddings, model_name=meta["model_name"])
