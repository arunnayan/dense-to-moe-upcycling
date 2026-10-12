"""Runs the whole experiment tree:

    Phase A : train dense -> checkpoint C
    Phase B : convert C -> MoE (exact clone), equivalence check
    Branches from C (same steps, same data order, fresh optimizer, same LR schedule):
        dense_cont : dense continued
        moe_clean  : exact-clone MoE, no noise          (primary)
        moe_noise  : exact-clone MoE + 1e-3 noise       (ablation)

Usage:  python run_all.py            (full run)
        python run_all.py --quick    (tiny smoke test, ~1 min)
Each finished stage is saved, so a re-run skips completed stages.
"""
import argparse
import copy
import json
import os

import torch

from data import CharData
from model import GPT, Config, count_params
from convert_to_moe import convert_to_moe, perturb_experts
from evaluate import equivalence_check, val_loss
from train import train

p = argparse.ArgumentParser()
p.add_argument("--quick", action="store_true")
args = p.parse_args()

S = dict(                        # all experiment settings in one place
    seed=0, batch_size=16, n_val_batches=20, val_batch_size=32,
    dense_steps=2000, dense_lr=1e-3, dense_min_lr=1e-4,
    cont_steps=1500, cont_lr=5e-4, cont_min_lr=5e-5, warmup=100,
    n_experts=8, top_k=2, aux_coef=0.01, noise_std=1e-3,
    eval_every=100, grad_every=25, diag_every=250,
)
if args.quick:
    S.update(dense_steps=60, cont_steps=60, warmup=10, n_val_batches=4,
             eval_every=20, grad_every=10, diag_every=20)

OUT = "results_quick" if args.quick else "results"
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(max(1, os.cpu_count() or 1))


def log(msg):
    print(msg, flush=True)
    with open(os.path.join(OUT, "log.txt"), "a") as f:
        f.write(msg + "\n")


def save(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1)


def done(name):
    return os.path.exists(os.path.join(OUT, name))


data = CharData(block_size=128)
cfg = Config(vocab_size=data.vocab_size)
val_batches = data.fixed_val_batches(S["n_val_batches"], S["val_batch_size"])
common = dict(batch_size=S["batch_size"], val_batches=val_batches, aux_coef=S["aux_coef"],
              eval_every=S["eval_every"], grad_every=S["grad_every"],
              diag_every=S["diag_every"], log=log)
save("settings.json", S)

# ---------------- Phase A: dense ----------------
ckpt = os.path.join(OUT, "dense_ckpt.pt")
if not os.path.exists(ckpt):
    torch.manual_seed(S["seed"])
    dense = GPT(cfg)
    log(f"dense params total/active: {count_params(dense)}")
    h = train(dense, data, steps=S["dense_steps"], peak_lr=S["dense_lr"],
              min_lr=S["dense_min_lr"], warmup=S["warmup"], seed=S["seed"],
              label="dense_A", **common)
    save("hist_dense_A.json", h)
    torch.save(dense.state_dict(), ckpt)
dense = GPT(cfg)
dense.load_state_dict(torch.load(ckpt))

# ---------------- Phase B: conversion + equivalence ----------------
moe_clean = convert_to_moe(dense, S["n_experts"], S["top_k"], router_seed=S["seed"])
eq = equivalence_check(dense, moe_clean, val_batches)
eq["dense_params"] = count_params(dense)
eq["moe_params_total_active"] = count_params(moe_clean)
moe_noise = copy.deepcopy(moe_clean)                 # same router init as moe_clean
perturb_experts(moe_noise, std=S["noise_std"], seed=S["seed"] + 1)
eq["noise_moe_val_loss"] = val_loss(moe_noise, val_batches)
save("equivalence.json", eq)
log(f"equivalence: {eq}")

# ---------------- Branches from checkpoint C ----------------
branches = {
    "dense_cont": copy.deepcopy(dense),
    "moe_clean": moe_clean,
    "moe_noise": moe_noise,
}
for name, model in branches.items():
    if done(f"hist_{name}.json"):
        continue
    h = train(model, data, steps=S["cont_steps"], peak_lr=S["cont_lr"],
              min_lr=S["cont_min_lr"], warmup=S["warmup"], seed=S["seed"] + 100,
              label=name, **common)
    save(f"hist_{name}.json", h)
    if name.startswith("moe"):
        torch.save(model.state_dict(), os.path.join(OUT, f"{name}_final.pt"))

log("ALL DONE")
