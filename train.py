"""One training loop used by every phase (dense, dense continuation, MoE).

Each call creates a FRESH AdamW optimizer with a warmup + cosine schedule, so
the dense-continuation and MoE branches start from identical optimizer
conditions (neither inherits Adam state from Phase A).
"""
import math
import time

import torch

from evaluate import val_loss, moe_diagnostics


def lr_at(step, steps, peak, floor, warmup):
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, steps - warmup)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * t))


def train(model, data, *, steps, peak_lr, min_lr, warmup, seed, batch_size,
          val_batches, aux_coef=0.01, eval_every=100, grad_every=25,
          diag_every=250, label="run", log=print):
    torch.manual_seed(seed)                              # dropout RNG
    gen = torch.Generator().manual_seed(seed)            # data order
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    is_moe = len(model.moe_layers()) > 0
    routers = [m.router.weight for m in model.moe_layers()]

    hist = {"label": label, "eval_step": [], "val_lm": [], "train_lm": [],
            "train_aux": [], "train_total": [], "grad_step": [],
            "router_grad_lm": [], "router_grad_total": [],
            "diag_step": [], "diag": []}

    def record_eval(step, window):
        hist["eval_step"].append(step)
        hist["val_lm"].append(val_loss(model, val_batches))
        if window:
            for k in ("lm", "aux", "total"):
                hist["train_" + k].append(sum(w[k] for w in window) / len(window))
        else:
            for k in ("lm", "aux", "total"):
                hist["train_" + k].append(None)
        if is_moe and step % diag_every == 0:
            hist["diag_step"].append(step)
            hist["diag"].append(moe_diagnostics(model, val_batches))
        model.train()

    record_eval(0, [])                                   # state BEFORE any update
    log(f"[{label}] step 0  val_lm {hist['val_lm'][-1]:.4f}")
    window, t0 = [], time.time()
    model.train()
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step - 1, steps, peak_lr, min_lr, warmup)
        x, y = data.train_batch(batch_size, gen)
        _, lm, aux = model(x, y)
        total = lm + aux_coef * aux

        opt.zero_grad(set_to_none=True)
        measure = is_moe and (step == 1 or step % grad_every == 0)
        if measure:
            # gradient of the LM loss ALONE w.r.t. the router weights
            g_lm = torch.autograd.grad(lm, routers, retain_graph=True)
            hist["grad_step"].append(step - 1)           # measured at weights of step-1
            hist["router_grad_lm"].append(
                math.sqrt(sum(g.pow(2).sum().item() for g in g_lm)))
        total.backward()
        if measure:
            hist["router_grad_total"].append(
                math.sqrt(sum(r.grad.pow(2).sum().item() for r in routers)))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        window.append({"lm": lm.item(), "aux": aux.item(), "total": total.item()})
        if step % eval_every == 0:
            record_eval(step, window)
            el = time.time() - t0
            log(f"[{label}] step {step}  train_lm {hist['train_lm'][-1]:.4f}  "
                f"val_lm {hist['val_lm'][-1]:.4f}  aux {hist['train_aux'][-1]:.4f}  "
                f"({el / step * 1000:.0f} ms/step)")
            window = []
    return hist
