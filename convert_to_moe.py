"""Dense -> MoE conversion (copy-upcycling) and the separate symmetry-breaking step.

These are deliberately two different functions:
  convert_to_moe()  : exact clone, must preserve the model's function
  perturb_experts() : tiny noise, applied ONLY AFTER the equivalence check
"""
import copy

import torch

from model import MoEFFN, DenseFFN


def convert_to_moe(dense_model, n_experts=8, top_k=2, router_seed=0):
    """Return a NEW model where every DenseFFN is replaced by an MoEFFN whose
    experts are exact copies of that layer's dense FFN. The dense model is untouched."""
    moe_model = copy.deepcopy(dense_model)
    torch.manual_seed(router_seed)                  # router init is reproducible
    for blk in moe_model.blocks:
        assert isinstance(blk.ffn, DenseFFN)
        dense_state = blk.ffn.ffn.state_dict()      # gate / up / down weights
        moe = MoEFFN(moe_model.cfg, n_experts=n_experts, top_k=top_k)
        for expert in moe.experts:
            expert.load_state_dict(dense_state)     # exact copy, bit-for-bit
        blk.ffn = moe
    return moe_model


@torch.no_grad()
def perturb_experts(moe_model, std=1e-3, seed=0):
    """Add N(0, std^2) noise to every expert weight (router untouched)."""
    gen = torch.Generator().manual_seed(seed)
    for moe in moe_model.moe_layers():
        for expert in moe.experts:
            for p in expert.parameters():
                p.add_(torch.randn(p.shape, generator=gen) * std)
