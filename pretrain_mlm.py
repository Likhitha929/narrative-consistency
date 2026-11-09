"""
pretrain_mlm.py — self-supervised pretraining of the NLI encoder on the novels.

The labelled set has only 1,669 distinct claims, which is far too little to
teach a randomly-initialised Transformer what the words in these books mean.
The books themselves are ~690k words of free, unlabelled text. Here the same
encoder used for classification is trained with BERT-style masked language
modelling (15% of tokens hidden; 80% [MASK] / 10% random / 10% unchanged)
on windows of the novels, laid out exactly like the fine-tuning input:

    [CLS] passage-like span A [SEP] following span B [SEP]
          segment 0                 segment 1

Nothing pretrained is used: the vocabulary comes from book_tokenizer.py and
all weights start random. The result (encoder.pt) initialises the
classifier in train_scratch.py.

    python pretrain_mlm.py --epochs 40
    python pretrain_mlm.py --epochs 40 --resume     # continue after interruption
"""

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import config
from book_tokenizer import book_files, load_tokenizer
from custom_nli_model import MLMHead, NarrativeNLIModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("pretrain")

# encoder size — shared with train_scratch.py through config.MODEL_ARCH
ARCH = config.MODEL_ARCH


def load_token_stream(tok):
    ids = []
    for f in book_files():
        text = Path(f).read_text(encoding="utf-8", errors="ignore")
        ids.extend(tok(text, add_special_tokens=False)["input_ids"])
        ids.append(tok.sep_token_id)
    return np.array(ids, dtype=np.int64)


def make_windows(stream, tok, max_len, rng, holdout=False):
    """Cut the stream into [CLS] A [SEP] B [SEP] windows starting at a random offset."""
    body = max_len - 3
    start = 0 if holdout else int(rng.integers(0, body))
    out = []
    for s in range(start, len(stream) - body, body):
        span = stream[s:s + body]
        a_len = int(rng.integers(int(body * 0.6), int(body * 0.85)))
        ids = np.concatenate([[tok.cls_token_id], span[:a_len], [tok.sep_token_id],
                              span[a_len:], [tok.sep_token_id]])
        tt = np.zeros(max_len, dtype=np.int64)
        tt[a_len + 2:] = 1
        out.append((ids, tt))
    return out


def mask_tokens(ids, tok, vocab_size, rng, p=0.15):
    ids = ids.copy()
    labels = np.full_like(ids, -100)
    special = (ids == tok.cls_token_id) | (ids == tok.sep_token_id) | (ids == tok.pad_token_id)
    pick = (rng.random(ids.shape) < p) & ~special
    labels[pick] = ids[pick]
    r = rng.random(ids.shape)
    ids[pick & (r < 0.8)] = tok.mask_token_id
    rand = pick & (r >= 0.8) & (r < 0.9)
    ids[rand] = rng.integers(5, vocab_size, rand.sum())
    return ids, labels


def batches(windows, tok, vocab_size, bs, rng, shuffle=True):
    order = rng.permutation(len(windows)) if shuffle else np.arange(len(windows))
    for i in range(0, len(order) - (bs - 1 if shuffle else 0), bs):
        chunk = [windows[j] for j in order[i:i + bs]]
        ids = np.stack([c[0] for c in chunk])
        tt = np.stack([c[1] for c in chunk])
        mids, lab = mask_tokens(ids, tok, vocab_size, rng)
        yield (torch.from_numpy(mids), torch.ones_like(torch.from_numpy(ids)),
               torch.from_numpy(tt), torch.from_numpy(lab))


@torch.no_grad()
def evaluate(model, head, windows, tok, vs):
    model.eval(); head.eval()
    rng = np.random.default_rng(0)                         # fixed masks -> comparable
    tot, n, correct = 0.0, 0, 0
    for ids, am, tt, lab in batches(windows, tok, vs, 64, rng, shuffle=False):
        m = lab != -100
        logits = head(model.encode(ids, am, tt)[m])         # only masked positions
        tot += F.cross_entropy(logits, lab[m], reduction="sum").item()
        correct += (logits.argmax(-1) == lab[m]).sum().item()
        n += m.sum().item()
    return tot / n, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=0, help="stop cleanly after N minutes (0 = no limit)")
    args = ap.parse_args()

    torch.manual_seed(config.SEED)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    out = Path(config.PRETRAIN_DIR); out.mkdir(parents=True, exist_ok=True)

    tok = load_tokenizer()
    vs = tok.vocab_size
    stream = load_token_stream(tok)
    n_hold = len(stream) // 50                          # last 2% of tokens held out
    train_stream, hold_stream = stream[:-n_hold], stream[-n_hold:]
    hold_windows = make_windows(hold_stream, tok, config.MAX_SEQ_LEN, np.random.default_rng(1), holdout=True)
    log.info(f"book tokens: {len(stream):,} | vocab {vs} | held-out windows {len(hold_windows)}")

    model = NarrativeNLIModel(vocab_size=vs, max_len=config.MAX_SEQ_LEN, **ARCH)
    head = MLMHead(model.embedding.token_embed)
    no_decay = lambda n_: n_.endswith("bias") or "norm" in n_
    named = [(f"m.{n_}", p) for n_, p in model.named_parameters()] + \
            [(f"h.{n_}", p) for n_, p in head.named_parameters() if not n_.startswith("embed.")]
    params = [p for _, p in named]
    opt = AdamW([{"params": [p for n_, p in named if not no_decay(n_)], "weight_decay": 0.01},
                 {"params": [p for n_, p in named if no_decay(n_)], "weight_decay": 0.0}],
                lr=args.lr, betas=(0.9, 0.98), eps=1e-6)

    steps_per_epoch = (len(train_stream) // (config.MAX_SEQ_LEN - 3)) // args.batch_size
    total = steps_per_epoch * args.epochs
    warm = max(1, int(0.06 * total))
    sched = LambdaLR(opt, lambda s: s / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))

    start_ep, hist = 1, []
    ck_path = out / "last_mlm.pt"
    if args.resume and ck_path.exists():
        ck = torch.load(ck_path, weights_only=False)
        model.load_state_dict(ck["model"]); head.load_state_dict(ck["head"])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_ep, hist = ck["epoch"] + 1, ck["hist"]
        log.info(f"resumed at epoch {start_ep}")

    log.info(f"encoder params: {model.count_parameters()[0]:,} | {steps_per_epoch} steps/epoch x {args.epochs}")
    t_start = time.time()
    for ep in range(start_ep, args.epochs + 1):
        model.train(); head.train()
        rng = np.random.default_rng(config.SEED + ep)
        windows = make_windows(train_stream, tok, config.MAX_SEQ_LEN, rng)
        t0, run = time.time(), []
        for ids, am, tt, lab in batches(windows, tok, vs, args.batch_size, rng):
            m = lab != -100
            logits = head(model.encode(ids, am, tt)[m])      # only masked positions
            loss = F.cross_entropy(logits, lab[m])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
            run.append(loss.item())
        hl, ha = evaluate(model, head, hold_windows, tok, vs)
        hist.append({"epoch": ep, "train_loss": float(np.mean(run)), "heldout_loss": hl,
                     "heldout_ppl": math.exp(hl), "heldout_mask_acc": ha})
        log.info(f"epoch {ep:02d}/{args.epochs} | train {np.mean(run):.3f} | held-out loss {hl:.3f} "
                 f"ppl {math.exp(hl):.1f} masked-acc {ha*100:.1f}% | {(time.time()-t0)/60:.1f} min")

        enc = {k: v for k, v in model.state_dict().items() if not k.startswith("nli_head")}
        torch.save(enc, out / "encoder.pt")
        (out / "encoder_config.json").write_text(json.dumps({"vocab_size": vs, "max_len": config.MAX_SEQ_LEN, **ARCH}, indent=2))
        (out / "pretrain_history.json").write_text(json.dumps(hist, indent=2))
        torch.save({"model": model.state_dict(), "head": head.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep, "hist": hist}, ck_path)
        if args.max_minutes and (time.time() - t_start) / 60 > args.max_minutes:
            log.info("time budget reached; re-run with --resume to continue")
            return
    log.info(f"done -> {out/'encoder.pt'}")


if __name__ == "__main__":
    main()
