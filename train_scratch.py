"""
train_scratch.py

Trains our from-scratch NarrativeNLIModel ONLY on the book dataset.
No SNLI, no pretrained weights. Everything starts random.
"""

import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import AutoTokenizer

from custom_nli_model import NarrativeNLIModel

logger = logging.getLogger(__name__)


# ─── Dataset ─────────────────────────────────────────────────────────────────

class BookDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len=256):
        self.samples   = samples
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # split "passage [SEP] claim" at the last [SEP]
        parts = s.input_text.rsplit(" [SEP] ", 1)
        passage = parts[0] if len(parts) == 2 else s.input_text
        claim   = parts[1] if len(parts) == 2 else ""

        enc = self.tokenizer(
            passage, claim,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get(
                "token_type_ids",
                torch.zeros(self.max_len, dtype=torch.long)
            ).squeeze(0),
            "labels": torch.tensor(s.label, dtype=torch.long),
        }


# ─── Helpers ─────────────────────────────────────────────────────────────────

def get_tokenizer(name_or_dir):
    """A model directory carries its own tokenizer (saved at train time);
    otherwise `name_or_dir` is a tokenizer directory or HF name."""
    from transformers import PreTrainedTokenizerFast
    d = Path(str(name_or_dir))
    if (d / "tokenizer.json").exists():
        return PreTrainedTokenizerFast.from_pretrained(str(d))
    return AutoTokenizer.from_pretrained(str(name_or_dir))


def group_balanced_batches(train_data, groups, batch_size, seed):
    """
    The CSV holds ~2.7 near-identical copies of every claim. Feeding all of
    them every epoch teaches the filler words and lets the model memorise
    each claim several times per epoch. Instead, each epoch draws ONE random
    copy per claim group, then interleaves the two classes so every batch is
    balanced.
    """
    rng = np.random.default_rng(seed)
    by_group = {}
    for i, g in enumerate(groups):
        by_group.setdefault(int(g), []).append(i)
    pick = [int(rng.choice(v)) for v in by_group.values()]
    c0 = [i for i in pick if train_data[i].label == 0]
    c1 = [i for i in pick if train_data[i].label == 1]
    rng.shuffle(c0); rng.shuffle(c1)
    order = []
    for k in range(max(len(c0), len(c1))):
        order.append(c0[k % len(c0)]); order.append(c1[k % len(c1)])
    return [order[k:k + batch_size] for k in range(0, len(order) - batch_size + 1, batch_size)]


def get_device():
    if torch.cuda.is_available():   return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, optimizer, scheduler, device, class_weights,
              training=True, accum_steps=1, rdrop_alpha=0.5):
    """
    One training or validation epoch.

    R-Drop (rdrop_alpha > 0, training only):
      Forward the same batch TWICE through the model (each time with different
      dropout masks). Penalise the symmetrical KL divergence between the two
      softmax distributions. This forces the model to produce stable/consistent
      outputs despite dropout noise — a strong regulariser for small datasets.

      Loss = task_loss + rdrop_alpha * KL(p1 || p2)
    """
    model.train() if training else model.eval()
    total_loss, steps = 0.0, 0

    ctx = torch.no_grad() if not training else torch.enable_grad()
    with ctx:
        for i, batch in enumerate(loader):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            ttid = batch["token_type_ids"].to(device)
            lbls = batch["labels"].to(device)

            out  = model(ids, mask, ttid, lbls, class_weights)
            loss = out["loss"]

            # ── R-Drop regularisation (training only) ───────────────────────
            if training and rdrop_alpha > 0:
                out2  = model(ids, mask, ttid, lbls, class_weights)   # 2nd forward pass
                p1    = torch.softmax(out["logits"],  dim=-1)
                p2    = torch.softmax(out2["logits"], dim=-1)
                # symmetric KL: 0.5*(KL(p1||p2) + KL(p2||p1))
                kl    = 0.5 * (
                    nn.functional.kl_div(p1.log(), p2, reduction="batchmean") +
                    nn.functional.kl_div(p2.log(), p1, reduction="batchmean")
                )
                loss  = loss + rdrop_alpha * kl

            loss = loss / accum_steps

            if training:
                loss.backward()
                if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # restored
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            total_loss += loss.item() * accum_steps
            steps += 1

    return total_loss / max(steps, 1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, trues = [], []
    for batch in loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        ttid = batch["token_type_ids"].to(device)
        out  = model(ids, mask, ttid)
        preds.extend(torch.argmax(out["logits"], dim=-1).cpu().tolist())
        trues.extend(batch["labels"].tolist())
    return preds, trues


@torch.no_grad()
def predict_with_probs(model, loader, device):
    """Returns (preds, trues, probs ndarray shape N×2)."""
    import numpy as np
    model.eval()
    preds, trues, probs_list = [], [], []
    for batch in loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        ttid = batch["token_type_ids"].to(device)
        out  = model(ids, mask, ttid)
        prob = torch.softmax(out["logits"], dim=-1)
        preds.extend(torch.argmax(out["logits"], dim=-1).cpu().tolist())
        trues.extend(batch["labels"].tolist())
        probs_list.append(prob.cpu())
    return preds, trues, torch.cat(probs_list, dim=0).numpy()


# ─── Main train function ──────────────────────────────────────────────────────

def train(
    train_data,
    val_data,
    save_dir=Path("model_scratch"),
    tokenizer_name="bert-base-uncased",
    batch_size=16,
    epochs=20,
    lr=2.5e-4,
    max_len=256,
    seed=42,
    accum_steps=2,
    patience=7,
    rdrop_alpha=0.35,
    resume=False,
    train_groups=None,
    pretrained_dir=None,
    arch=None,
    finetune_dropout=0.40,
    focal_gamma=2.0,
    label_smoothing=0.15,
    weight_decay=0.10,
):
    """
    Train on `train_data`, early-stop / checkpoint on `val_data`.
    The split is made by the caller (main.py → splits.grouped_split) so that
    paraphrased copies of a claim never appear on both sides.

    train_groups   : claim-group id per training row -> one copy per claim per epoch
    pretrained_dir : directory with encoder.pt from pretrain_mlm.py (optional)
    arch           : encoder size dict (must match the pretrained encoder)

    Checkpointing keeps the epoch with the best val macro-F1 (ties -> lower val loss).

    Every epoch writes `save_dir/last.pt` (model + optimiser + scheduler +
    history), so an interrupted run can continue with resume=True.
    """
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]      = "1"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_device()
    logger.info(f"device: {device}")

    # tokenizer — vocabulary mapping only, zero model weights loaded
    logger.info(f"loading tokenizer vocabulary: {tokenizer_name}")
    tokenizer = get_tokenizer(tokenizer_name)

    train_ratio = sum(s.label for s in train_data) / len(train_data)
    val_ratio   = sum(s.label for s in val_data)   / len(val_data)
    logger.info(f"train consistent ratio: {train_ratio:.2f} | val consistent ratio: {val_ratio:.2f}")

    train_ds = BookDataset(train_data, tokenizer, max_len)
    val_ds   = BookDataset(val_data,   tokenizer, max_len)

    # ── class weights ────────
    counts = np.bincount([s.label for s in train_data], minlength=2)
    total  = len(train_data)
    w0 = total / (2 * counts[0]) if counts[0] > 0 else 1.0
    w1 = total / (2 * counts[1]) if counts[1] > 0 else 1.0

    class_weights = torch.tensor([w0, w1], dtype=torch.float).to(device)

    # ── exact batch stratification  ───────────
    contradict_idx = [i for i, s in enumerate(train_data) if s.label == 0]
    consistent_idx = [i for i, s in enumerate(train_data) if s.label == 1]

    interleaved = []
    for i in range(max(len(contradict_idx), len(consistent_idx))):
        interleaved.append(contradict_idx[i % len(contradict_idx)])
        interleaved.append(consistent_idx[i % len(consistent_idx)])

    from torch.utils.data import BatchSampler
    if train_groups is not None:
        logger.info(f"group-balanced sampling: {len(set(map(int, train_groups)))} claims, one copy each per epoch")
        make_train_dl = lambda ep: DataLoader(
            train_ds, batch_sampler=group_balanced_batches(train_data, train_groups, batch_size, seed + ep),
            num_workers=0)
    else:
        make_train_dl = lambda ep: DataLoader(
            train_ds, batch_sampler=BatchSampler(interleaved, batch_size, drop_last=True), num_workers=0)
    train_dl = make_train_dl(0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info(f"train: {len(train_ds)} | val: {len(val_ds)}")
    logger.info(f"class weights — contradict: {w0:.3f}, consistent: {w1:.3f}")
    logger.info(f"stratified batches — contradict: {len(contradict_idx)} | consistent: {len(consistent_idx)}")

    # ── build model from scratch ──────────────────────────────────────────────
    logger.info("building NarrativeNLIModel from scratch (all weights random)...")
    arch = dict(arch or dict(embed_dim=256, num_heads=8, num_layers=4, ffn_dim=512))
    arch["dropout"] = finetune_dropout
    model = NarrativeNLIModel(
        vocab_size      = tokenizer.vocab_size,
        max_len         = max_len,
        num_labels      = 2,
        focal_gamma     = focal_gamma,
        label_smoothing = label_smoothing,
        **arch,
    )
    if pretrained_dir is not None:
        model.load_encoder(str(pretrained_dir))
    model = model.to(device)

    total_p, train_p = model.count_parameters()
    logger.info(f"model params: {total_p:,} total | {train_p:,} trainable")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)  
    total_steps = (len(train_dl) // accum_steps + 1) * epochs
    warmup_steps = max(1, int(0.10 * total_steps))  

    def lr_lambda(step):
        """Linear warmup then cosine decay to 5% of peak LR."""
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # ── training loop ─────────────────────────────────────────────────────────
    best_f1       = 0.0
    best_val_loss = float("inf")
    no_improve    = 0
    id2label      = {0: "contradict", 1: "consistent"}
    save_dir      = Path(save_dir)
    history       = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    start_epoch   = 1
    last_ckpt     = save_dir / "last.pt"

    if resume and last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        history       = ck["history"]
        best_f1       = ck["best_f1"]
        best_val_loss = ck["best_val_loss"]
        no_improve    = ck["no_improve"]
        start_epoch   = ck["epoch"] + 1
        torch.set_rng_state(ck["torch_rng"])
        np.random.set_state(ck["np_rng"])
        logger.info(f"resumed from {last_ckpt} at epoch {start_epoch} (best_f1={best_f1:.4f})")

    stopped_early = no_improve >= patience
    for epoch in range(start_epoch, epochs + 1):
        if stopped_early:
            break
        t0 = time.time()

        train_dl = make_train_dl(epoch)
        train_loss = run_epoch(model, train_dl, optimizer, scheduler, device,
                               class_weights, training=True, accum_steps=accum_steps,
                               rdrop_alpha=rdrop_alpha)

        # compute val loss — key for overfitting detection
        val_loss_ep = run_epoch(model, val_dl, None, None, device,
                                class_weights, training=False, accum_steps=1)
        val_preds, val_true = predict(model, val_dl, device)

        acc     = accuracy_score(val_true, val_preds)
        f1      = f1_score(val_true, val_preds, average="macro", zero_division=0)
        gap     = val_loss_ep - train_loss
        elapsed = (time.time() - t0) / 60

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss_ep)
        history["val_acc"].append(acc)
        history["val_f1"].append(f1)

        gap_warn = "" if gap > 0.08 else ""
        logger.info(
            f"epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss_ep:.4f} | "
            f"gap={gap:+.4f}{gap_warn} | "
            f"val_acc={acc:.4f} | val_f1={f1:.4f} | {elapsed:.1f}min"
        )

        # ── checkpoint on best val macro-F1 (ties -> lower val loss) ──────
        improved = f1 > best_f1 + 1e-4 or (abs(f1 - best_f1) <= 1e-4 and val_loss_ep < best_val_loss)

        if improved:
            best_f1       = f1
            best_val_loss = val_loss_ep
            no_improve    = 0
            model.save(str(save_dir))
            tokenizer.save_pretrained(str(save_dir))
            logger.info(f"  ✓ saved best (f1={f1:.4f}, val_loss={val_loss_ep:.4f})")
        else:
            no_improve += 1
            logger.info(f"  no improvement ({no_improve}/{patience}) | best_f1={best_f1:.4f}")
            if no_improve >= patience:
                logger.info("  early stopping triggered.")
                stopped_early = True

        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "history": history,
            "best_f1": best_f1, "best_val_loss": best_val_loss,
            "no_improve": no_improve, "epoch": epoch,
            "torch_rng": torch.get_rng_state(), "np_rng": np.random.get_state(),
        }, last_ckpt)

    # ── final report ──────────────────────────────────────────────────────────
    best_model = NarrativeNLIModel.load(str(save_dir)).to(device)
    final_preds, final_true, final_probs = predict_with_probs(best_model, val_dl, device)
    names = [id2label[i] for i in sorted(id2label)]
    logger.info(f"\n{classification_report(final_true, final_preds, target_names=names, zero_division=0)}")
    logger.info(f"best val F1: {best_f1:.4f}")

    # persist the curves so `main.py --skip-training` can still plot them
    import json
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # return model, tokenizer, training history, and val loader for downstream use
    return best_model, tokenizer, {
        "train_loss":  history["train_loss"],
        "val_loss":    history["val_loss"],
        "val_acc":     history["val_acc"],
        "val_f1":      history["val_f1"],
        "val_preds":   final_preds,
        "val_true":    final_true,
        "val_probs":   final_probs,
        "val_samples": val_data,
    }, val_dl
