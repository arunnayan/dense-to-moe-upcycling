"""Evaluation and diagnostics. Everything runs in eval mode, no gradients,
on the fixed validation batches."""
import itertools

import torch
import torch.nn.functional as F


@torch.no_grad()
def val_loss(model, val_batches):
    model.eval()
    losses = [model(x, y)[1].item() for x, y in val_batches]
    return sum(losses) / len(losses)


@torch.no_grad()
def equivalence_check(dense_model, moe_model, val_batches):
    """Max |logit difference| and val loss of both models on identical inputs."""
    dense_model.eval()
    moe_model.eval()
    max_diff = 0.0
    for x, _ in val_batches:
        d = dense_model(x)[0]
        m = moe_model(x)[0]
        max_diff = max(max_diff, (d - m).abs().max().item())
    return {
        "max_abs_logit_diff": max_diff,
        "dense_val_loss": val_loss(dense_model, val_batches),
        "moe_val_loss": val_loss(moe_model, val_batches),
    }


def _mean_pairwise_cos(vectors):
    """vectors: (E, D). Mean cosine similarity over all E*(E-1)/2 pairs."""
    v = F.normalize(vectors.float(), dim=-1)
    sims = [(v[i] * v[j]).sum().item() for i, j in itertools.combinations(range(len(v)), 2)]
    return sum(sims) / len(sims)


@torch.no_grad()
def moe_diagnostics(moe_model, val_batches, n_tokens_for_output_sim=2048):
    """Per layer: expert load share, load CV, weight cosine, output cosine."""
    moe_model.eval()
    layers = moe_model.moe_layers()

    # 1) expert load over all val batches, and capture each layer's FFN input
    loads = [torch.zeros(m.n_experts) for m in layers]
    captured = [[] for _ in layers]
    hooks = [m.register_forward_pre_hook(
        lambda mod, inp, i=i: captured[i].append(inp[0].reshape(-1, inp[0].shape[-1])))
        for i, m in enumerate(layers)]
    for x, _ in val_batches:
        moe_model(x)
        for i, m in enumerate(layers):
            loads[i] += m.last_load
    for h in hooks:
        h.remove()

    out = {"load_share": [], "load_cv": [], "weight_cos": [], "output_cos": []}
    for i, m in enumerate(layers):
        share = loads[i] / loads[i].sum()
        out["load_share"].append([round(s, 4) for s in share.tolist()])
        out["load_cv"].append((share.std(unbiased=False) / share.mean()).item())

        # 2) weight similarity: flatten all weights of each expert into one vector
        flat = torch.stack([torch.cat([p.flatten() for p in e.parameters()]) for e in m.experts])
        out["weight_cos"].append(_mean_pairwise_cos(flat))

        # 3) output similarity: run EVERY expert on the same tokens, compare outputs
        xs = torch.cat(captured[i])[:n_tokens_for_output_sim]
        outs = torch.stack([e(xs) for e in m.experts])            # (E, N, C)
        outs = F.normalize(outs, dim=-1)
        sims = [(outs[a] * outs[b]).sum(-1).mean().item()
                for a, b in itertools.combinations(range(m.n_experts), 2)]
        out["output_cos"].append(sum(sims) / len(sims))

    for k in ("load_cv", "weight_cos", "output_cos"):
        out[k + "_mean"] = sum(out[k]) / len(out[k])
    return out
