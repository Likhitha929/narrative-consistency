import logging
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class ClassificationSample:
    sample_id:  int
    input_text: str
    label:      int   # -1 for test (no ground truth)
    book_name:  str
    claim:      str = ""   # raw claim text (used for leakage-free grouping)


def build_samples(df, retriever, top_k=3, label2id=None, is_test=False, with_char=True,
                  evidence_tokenizer=None, max_len=256):
    """evidence_tokenizer given -> sentence-level evidence packed into max_len (evidence.py);
    otherwise the top_k whole passages are concatenated (original behaviour)."""
    if label2id is None:
        label2id = {"consistent": 1, "contradict": 0}

    samples = []

    for _, row in df.iterrows():
        claim     = str(row["content"]).strip()
        book_name = str(row["book_name"]).strip()
        row_id    = int(row["id"]) if str(row["id"]).isdigit() else 0

        if not claim:
            continue

        # Two thirds of the claims open with "He"/"She" and only ~15% name the
        # character, so the bare claim is a poor search query. Prefix the
        # character from the `char` column to both the query and the claim.
        char = str(row.get("char", "") or "").strip()
        if char.lower() == "nan":
            char = ""
        subject_claim = f"{char}: {claim}" if (with_char and char) else claim

        # get top-k passages relevant to this claim from the correct book
        if evidence_tokenizer is not None:
            from evidence import select_evidence
            budget = max_len - 3 - len(evidence_tokenizer(subject_claim, add_special_tokens=False)["input_ids"])
            ev = select_evidence(retriever, subject_claim, book_name, char, evidence_tokenizer, budget)
            combined = ev + " [SEP] " + subject_claim
        else:
            retrieved = retriever.retrieve(query=subject_claim, top_k=top_k, book_name=book_name)
            # build input: passage1 [SEP] passage2 [SEP] ... [SEP] claim
            passage_texts = [r.passage.text for r in retrieved]
            combined = " [SEP] ".join(passage_texts + [subject_claim])

        # resolve the label
        if is_test:
            label = -1
        else:
            raw = str(row.get("label", "")).strip().lower()
            if raw not in label2id:
                logger.warning(f"unknown label '{raw}' at id={row_id}, skipping")
                continue
            label = label2id[raw]

        samples.append(ClassificationSample(
            sample_id=row_id,
            input_text=combined,
            label=label,
            book_name=book_name,
            claim=claim,
        ))

    logger.info(f"built {len(samples)} samples (is_test={is_test})")
    return samples


class NarrativeDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=512):
        self.samples    = samples
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        enc = self.tokenizer(
            s.input_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

        if s.label != -1:
            item["labels"] = torch.tensor(s.label, dtype=torch.long)

        return item
