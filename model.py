"""A small GPT-style Transformer whose FFN can be dense or Mixture-of-Experts.

Block layout (pre-LayerNorm):
    x = x + Attention(LN1(x))
    x = x + FFN(LN2(x))        <- FFN is DenseFFN or MoEFFN

Only the FFN changes during conversion. Attention, embeddings and LayerNorms
are carried over untouched.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 65
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    d_ff: int = 512          # hidden width of the SwiGLU FFN
    dropout: float = 0.1


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class SwiGLU(nn.Module):
    """FFN(x) = down( silu(gate(x)) * up(x) ). Used as the dense FFN AND as each expert."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DenseFFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.ffn(x)), x.new_zeros(())   # (output, aux loss = 0)


class MoEFFN(nn.Module):
    """Top-k routed experts.

    Three properties the equivalence test depends on:
      1. top-k weights are RENORMALISED so they sum to 1 for every token
      2. DROPLESS: every token is processed by its k experts, no capacity limit
      3. dropout only acts in train mode (the check runs in eval mode)
    """

    def __init__(self, cfg, n_experts=8, top_k=2, router_init_scale=0.1):
        super().__init__()
        self.n_experts, self.top_k = n_experts, top_k
        self.router = nn.Linear(cfg.d_model, n_experts, bias=False)
        with torch.no_grad():                       # small router init (lecture: x 1/10)
            self.router.weight.mul_(router_init_scale)
        self.experts = nn.ModuleList(SwiGLU(cfg.d_model, cfg.d_ff) for _ in range(n_experts))
        self.drop = nn.Dropout(cfg.dropout)
        self.last_load = None                       # assignments per expert, last forward

    def forward(self, x):
        B, T, C = x.shape
        xf = x.reshape(-1, C)                                   # (N, C), N = B*T tokens
        probs = F.softmax(self.router(xf).float(), dim=-1)      # (N, E)
        top_p, top_i = probs.topk(self.top_k, dim=-1)           # (N, k)
        top_w = top_p / top_p.sum(dim=-1, keepdim=True)         # renormalise: sums to 1

        out = torch.zeros_like(xf)
        for e, expert in enumerate(self.experts):
            tok, slot = (top_i == e).nonzero(as_tuple=True)     # tokens that picked e
            if tok.numel() == 0:
                continue
            y = expert(xf[tok]) * top_w[tok, slot].unsqueeze(-1).to(xf.dtype)
            out.index_add_(0, tok, y)

        # Switch-Transformer balancing loss: E * sum_i f_i * P_i  (=1.0 when perfectly even)
        counts = torch.bincount(top_i.flatten(), minlength=self.n_experts).float()
        f = counts / counts.sum()                   # fraction of assignments (no gradient)
        P = probs.mean(dim=0)                       # mean router probability (has gradient)
        aux = self.n_experts * (f * P).sum()
        self.last_load = counts.detach()
        return self.drop(out.view(B, T, C)), aux


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = DenseFFN(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        h, aux = self.ffn(self.ln2(x))
        return x + h, aux


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight      # weight tying
        self.apply(self._init)
        for n, p in self.named_parameters():        # GPT-2 style scaled residual init
            if n.endswith("proj.weight") or n.endswith("down.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        aux_total = x.new_zeros(())
        for blk in self.blocks:
            x, aux = blk(x)
            aux_total = aux_total + aux
        logits = self.head(self.ln_f(x))
        lm_loss = None
        if targets is not None:
            lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, lm_loss, aux_total

    def moe_layers(self):
        return [b.ffn for b in self.blocks if isinstance(b.ffn, MoEFFN)]


def count_params(model):
    """(total, active-per-token). Active counts only top_k of the experts in each MoE layer."""
    seen, total = set(), 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    active = total
    for moe in model.moe_layers():
        per_expert = sum(p.numel() for p in moe.experts[0].parameters())
        active -= per_expert * (moe.n_experts - moe.top_k)
    return total, active
