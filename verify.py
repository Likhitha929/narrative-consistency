"""
verify.py — check one claim against a book, end to end.

    from verify import Verifier
    v = Verifier()
    r = v.check("Edmond Dantès", "He escaped from the Chateau d'If hidden in a burial sack.",
                "The Count of Monte Cristo")
    r["verdict"], r["p_consistent"], r["evidence"]

Used by app.py (the Streamlit demo) and handy from a notebook.
"""

from pathlib import Path

import torch

import config
import embedding
import retrieval
from custom_nli_model import NarrativeNLIModel
from evidence import select_evidence
from train_scratch import get_tokenizer

BOOKS = ["The Count of Monte Cristo", "In Search of the Castaways"]


ENSEMBLE_DIR = Path("models_ensemble")


class Verifier:
    """Averages the seed ensemble in models_ensemble/ when present, else uses model_dir."""

    def __init__(self, model_dir=None, index_dir=Path("cache/embedding_index")):
        if model_dir is None:
            dirs = sorted(ENSEMBLE_DIR.glob("seed_*")) or [config.MODEL_SAVE_DIR]
        else:
            dirs = [Path(model_dir)]
        self.tok = get_tokenizer(dirs[0])
        self.models = [NarrativeNLIModel.load(str(d)).eval() for d in dirs]
        index_dir = Path(index_dir)
        if (index_dir / "embeddings.npy").exists():
            index = embedding.load_index(index_dir)
        else:   # first run on a fresh clone without the shipped index: build it from Dataset/Books
            import chunking, data_loader
            books = data_loader.load_books(config.BOOKS_DIR)
            passages = chunking.chunk_all_books(books, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            index = embedding.build_embedding_index(passages, config.EMBEDDING_MODEL_NAME, batch_size=64)
            embedding.save_index(index, index_dir)
        self.retriever = retrieval.Retriever(index)

    @torch.no_grad()
    def _p_consistent(self, evidence, claim):
        enc = self.tok(evidence, claim, max_length=config.MAX_SEQ_LEN, truncation=True,
                       padding="max_length", return_tensors="pt")
        ps = [torch.softmax(m(enc["input_ids"], enc["attention_mask"], enc["token_type_ids"])["logits"], -1)[0, 1]
              for m in self.models]
        return float(torch.stack(ps).mean())

    def check(self, character, claim, book):
        subject_claim = f"{character}: {claim}" if character else claim
        budget = config.MAX_SEQ_LEN - 3 - len(self.tok(subject_claim, add_special_tokens=False)["input_ids"])
        evidence = select_evidence(self.retriever, subject_claim, book, character, self.tok, budget)
        p = self._p_consistent(evidence, subject_claim)
        p_blank = self._p_consistent("", subject_claim)       # same claim, no evidence
        return {"verdict": "consistent" if p >= 0.5 else "contradict",
                "p_consistent": p, "p_without_evidence": p_blank, "evidence": evidence}


if __name__ == "__main__":
    v = Verifier()
    for ch, cl in [("Edmond Dantès", "He escaped from the Chateau d'If by taking the place of a dead prisoner in a burial sack."),
                   ("Edmond Dantès", "He escaped from the Chateau d'If by bribing the governor with the treasure of Monte Cristo.")]:
        r = v.check(ch, cl, BOOKS[0])
        print(f"{r['verdict']:>10}  p={r['p_consistent']:.2f}  (no evidence {r['p_without_evidence']:.2f})  {cl}")
