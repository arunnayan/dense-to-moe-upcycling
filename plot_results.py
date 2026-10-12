"""Make the three plots + a summary table from the saved histories.

Usage: python plot_results.py [results_dir]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "results"


def load(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


A = load("hist_dense_A.json")
H = {k: load(f"hist_{k}.json") for k in ("dense_cont", "moe_clean", "moe_noise")}
eq = load("equivalence.json")
C = A["eval_step"][-1]                       # conversion step on the global axis
COL = {"dense_A": "#444444", "dense_cont": "#1f77b4", "moe_clean": "#d62728", "moe_noise": "#ff9f1c"}
NAME = {"dense_A": "Dense (Phase A)", "dense_cont": "Dense continued (control)",
        "moe_clean": "MoE, exact clone", "moe_noise": "MoE, clone + 1e-3 noise"}
shift = lambda steps: [C + s for s in steps]

# ---------------- Plot 1: main result (LM val loss only) ----------------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for a in ax:
    a.plot(A["eval_step"], A["val_lm"], color=COL["dense_A"], label=NAME["dense_A"])
    for k, h in H.items():
        a.plot(shift(h["eval_step"]), h["val_lm"], color=COL[k], label=NAME[k],
               ls="--" if k == "moe_noise" else "-")
    a.axvline(C, color="grey", ls=":", lw=1)
    a.set_xlabel("training step")
    a.set_ylabel("validation LM loss")
    a.grid(alpha=.3)
ax[0].set_title("Validation LM loss, whole run")
ax[0].text(C, ax[0].get_ylim()[1] * 0.97, " conversion", color="grey", va="top")
ax[0].legend()
lo = min(min(h["val_lm"]) for h in H.values())
hi = max(A["val_lm"][-4:])
ax[1].set_xlim(C - 4 * (A["eval_step"][1] - A["eval_step"][0]), C + H["moe_clean"]["eval_step"][-1])
ax[1].set_ylim(lo - 0.02, hi + 0.03)
ax[1].set_title("Zoom: around and after conversion")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "plot1_main_result.png"), dpi=130)

# ---------------- Plot 2: MoE objective: LM vs aux vs total ----------------
coef = load("settings.json")["aux_coef"]
m = H["moe_clean"]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
st, lm = m["eval_step"][1:], m["train_lm"][1:]
aux, tot = m["train_aux"][1:], m["train_total"][1:]
ax[0].plot(A["eval_step"][1:], A["train_total"][1:], color=COL["dense_A"],
           label="dense total (= LM)")
ax[0].plot(shift(st), lm, color=COL["moe_clean"], label="MoE LM loss")
ax[0].plot(shift(st), tot, color="purple", ls="--", label=f"MoE total = LM + {coef} x aux")
ax[0].axvline(C, color="grey", ls=":", lw=1)
ax[0].set_xlim(C - 1000, C + st[-1])
ax[0].set_ylim(min(lm) - 0.03, max(v for v in A["train_total"][-10:] if v is not None) + 0.06)
ax[0].set_title("Train loss (window mean): total jumps, LM does not")
ax[0].set_xlabel("training step"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(shift(st), aux, color="purple")
ax[1].axhline(len(m["diag"][0]["load_cv"]), color="grey", ls=":", lw=1,
              label="perfect balance (1.0 per layer)")
ax[1].set_title("Aux balancing loss (sum over 4 MoE layers)")
ax[1].set_xlabel("training step"); ax[1].legend(); ax[1].grid(alpha=.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "plot2_moe_objective.png"), dpi=130)

# ---------------- Plot 3: MoE diagnostics ----------------
fig, ax = plt.subplots(2, 3, figsize=(15, 8.2))
for k in ("moe_clean", "moe_noise"):
    h = H[k]
    ls = "--" if k == "moe_noise" else "-"
    ax[0, 0].plot(h["diag_step"], [d["load_cv_mean"] for d in h["diag"]], color=COL[k], ls=ls, label=NAME[k])
    ax[0, 2].plot(h["diag_step"], [max(0.0, 1 - d["weight_cos_mean"]) for d in h["diag"]], color=COL[k], ls=ls, label=NAME[k])
    ax[1, 0].plot(h["diag_step"], [max(0.0, 1 - d["output_cos_mean"]) for d in h["diag"]], color=COL[k], ls=ls, label=NAME[k])
ax[0, 0].set_title("Load imbalance: CV = std / mean of expert share")
share = H["moe_clean"]["diag"][-1]["load_share"]
im = ax[0, 1].imshow(share, aspect="auto", cmap="viridis", vmin=0)
ax[0, 1].set_title("Final expert load share (MoE, exact clone)")
ax[0, 1].set_xlabel("expert"); ax[0, 1].set_ylabel("layer")
ax[0, 1].set_yticks(range(len(share))); ax[0, 1].set_xticks(range(len(share[0])))
for i, row in enumerate(share):
    for j, v in enumerate(row):
        ax[0, 1].text(j, i, f"{100 * v:.1f}", ha="center", va="center", color="w", fontsize=8)
fig.colorbar(im, ax=ax[0, 1])
ax[0, 2].set_title("Weight divergence: 1 - mean pairwise cosine")
ax[1, 0].set_title("Output divergence: 1 - mean pairwise cosine")
for a in (ax[0, 2], ax[1, 0]):
    a.set_yscale("symlog", linthresh=1e-6)
for j, k in enumerate(("moe_clean", "moe_noise")):
    h, a = H[k], ax[1, 1 + j]
    a.plot(h["grad_step"], [max(g, 1e-12) for g in h["router_grad_lm"]], label="||grad router L_LM||")
    a.plot(h["grad_step"], [max(g, 1e-12) for g in h["router_grad_total"]], label="||grad router L_total||")
    a.set_yscale("log")
    a.set_title(f"Router gradient norms: {NAME[k]}")
for a in ax.flat:
    if a is not ax[0, 1]:
        a.set_xlabel("steps after conversion"); a.grid(alpha=.3); a.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "plot3_moe_diagnostics.png"), dpi=130)

# ---------------- Summary table ----------------
rows = {"dense_A": {"val_start": A["val_lm"][0], "val_end": A["val_lm"][-1]}}
for k, h in H.items():
    r = {"val_start": h["val_lm"][0], "val_end": h["val_lm"][-1],
         "val_min": min(h["val_lm"])}
    if h["diag"]:
        d0, d1 = h["diag"][0], h["diag"][-1]
        r.update(load_cv_start=d0["load_cv_mean"], load_cv_end=d1["load_cv_mean"],
                 weight_cos_start=d0["weight_cos_mean"], weight_cos_end=d1["weight_cos_mean"],
                 output_cos_start=d0["output_cos_mean"], output_cos_end=d1["output_cos_mean"],
                 router_grad_lm_first=h["router_grad_lm"][0],
                 router_grad_total_first=h["router_grad_total"][0],
                 router_grad_lm_last=h["router_grad_lm"][-1])
    rows[k] = r
summary = {"equivalence": eq, "runs": rows}
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=1)
print(json.dumps(summary, indent=1))
