"""Figures for report.pdf (static PNGs)."""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent
BLUE, ORANGE, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.dpi": 220})

# 1. pretraining curves (two panels, no dual axis)
h = json.load(open("model_pretrained/pretrain_history.json"))
ep = [r["epoch"] for r in h]
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.3))
for a, key, f, title in [(ax[0], "heldout_ppl", 1, "Held-out perplexity (lower is better)"),
                         (ax[1], "heldout_mask_acc", 100, "Held-out masked-token accuracy (%)")]:
    a.plot(ep, [r[key] * f for r in h], color=BLUE, lw=2)
    a.set_title(title, color=INK, fontsize=9, loc="left"); a.set_xlabel("pretraining epoch")
    a.grid(axis="y", color=GRID, lw=.6)
    for e in (30, 60, 100):
        v = h[e - 1][key] * f
        a.plot(e, v, "o", ms=5, color=BLUE, mec=SURF, mew=1.5)
        a.annotate(f"{v:.0f}" if f == 1 else f"{v:.1f}", (e, v), textcoords="offset points",
                   xytext=(0, 7), ha="center", color=INK, fontsize=8)
ax[0].set_ylim(0, None); ax[1].set_ylim(0, 33)
fig.tight_layout(); fig.savefig(OUT / "fig_pretraining.png"); plt.close(fig)

# 2. progression of held-out test accuracy
F = json.load(open("exp/final_eval.json"))
stages = [("Control: no pretraining (random init)", 45.6), ("Original model (BERT vocab)", 63.6),
          ("+ book tokenizer, MLM 30 ep, synthetic", 63.6), ("+ MLM 60 epochs", 65.4),
          ("+ sentence-level evidence", 68.1), ("+ MLM 100 epochs (3-seed ensemble)", F["ensemble"]["test"]["acc"] * 100)]
fig, a = plt.subplots(figsize=(6.6, 2.6))
y = list(range(len(stages)))[::-1]
a.barh(y, [s[1] for s in stages], color=BLUE, height=.55)
for yi, (n, v) in zip(y, stages):
    a.text(v + .6, yi, f"{v:.1f}%", va="center", color=INK, fontsize=8, bbox=dict(fc=SURF, ec="none", pad=1), zorder=5)
b = F["baseline_claim"]["acc"] * 100
a.axvline(b, color=INK2, ls="--", lw=1); a.text(b + .6, y[2], f"claim-only\nbaseline\n{b:.1f}%", color=INK2, fontsize=7.5, va="center")
a.axvline(55.7, color=INK2, ls=":", lw=.8); a.text(55.9, y[0] + .45, "majority 55.7%", color=INK2, fontsize=7)
a.set_yticks(y); a.set_yticklabels([s[0] for s in stages]); a.set_xlim(40, 80); a.set_ylim(-.5, len(stages) - .1)
a.set_xlabel("held-out test accuracy (%), n = 653 unseen claims")
fig.tight_layout(); fig.savefig(OUT / "fig_progression.png"); plt.close(fig)

# 3. evidence ablation
names = ["seed 42", "seed 7", "seed 123", "ensemble"]
seeds = list(F["seeds"].values()) + [F["ensemble"]]
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.4))
for a, key, f, title in [(ax[0], "acc", 100, "Accuracy (%)"), (ax[1], "auc", 1, "ROC-AUC")]:
    w = .36; x = range(len(names))
    a.bar([i - w/2 - .01 for i in x], [s["test"][key] * f for s in seeds], w, color=BLUE, label="with evidence")
    a.bar([i + w/2 + .01 for i in x], [s["blank"][key] * f for s in seeds], w, color=ORANGE, label="evidence removed")
    a.set_xticks(list(x)); a.set_xticklabels(names); a.set_title(title, color=INK, fontsize=9, loc="left")
    a.set_ylim((50, 76) if f == 100 else (.55, .82)); a.grid(axis="y", color=GRID, lw=.6); a.set_axisbelow(True)
fig.legend(*ax[0].get_legend_handles_labels(), frameon=False, fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, .92)); fig.savefig(OUT / "fig_ablation.png"); plt.close(fig)
print("figures ->", OUT)
