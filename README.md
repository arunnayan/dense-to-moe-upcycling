# Dense → MoE: train a dense Transformer, convert it, keep training

ERA V5 Session 14 assignment:

> Train a Linear model and convert that into an MoE! Your call on model size
> and data trained on, but must show they continue to train and reduce loss!

"Linear model" is read as a **dense** Transformer (every token goes through
the same FFN). The FFN in every block is then converted into a Mixture of
Experts by copy-upcycling, and training continues.

Every number below comes from `results/` produced by `python run_all.py` on
a single CPU core. Nothing is hand-entered.

---

## Experiment tree

```
                    TinyShakespeare (char-level, vocab 65)
                              |
                 Dense Transformer, 2,000 steps          (Phase A)
                              |
                        checkpoint C
                       /            \
          Dense continued          Dense -> MoE (exact clone of every FFN)
          1,500 steps                     |
          (control)               equivalence check
                                   /              \
                        MoE, exact clone     MoE, clone + 1e-3 noise
                        1,500 steps          1,500 steps
                        (primary)            (ablation)
```

All three branches start from checkpoint C with the same data order, the same
seed, a **fresh** AdamW optimizer and the same LR schedule (100-step warmup to
5e-4, cosine to 5e-5). The only difference is the FFN.

## Setup

| | Dense | MoE |
|---|---|---|
| Layers / d_model / heads / context | 4 / 128 / 4 / 128 | same |
| FFN | SwiGLU, hidden 512 | 8 SwiGLU experts, hidden 512 each |
| Routing | – | top-2, renormalised, dropless |
| Router init | – | default x 0.1 |
| Balancing | – | Switch aux loss `E * sum(f_i * P_i)`, coefficient 0.01 |
| Total params | 1,075,584 | 6,584,704 |
| Active params / token | 1,075,584 | 1,866,112 |

Batch 16 x 128 tokens, AdamW (betas 0.9/0.95, weight decay 0.1), grad clip 1.0,
dropout 0.1. Validation uses the same 20 fixed batches (20 x 32 x 128 tokens)
every time.

---

## Results

### Q1. Does the dense model learn?

| | val LM loss |
|---|---|
| Phase A, step 0 | 4.2046 |
| Phase A, step 2,000 (checkpoint C) | 1.8179 |

Yes.

### Q2. Does conversion preserve what it learned?

Measured in eval mode on the fixed val batches:

| | value |
|---|---|
| Max \|dense logit − MoE logit\| | 1.43e-5 |
| Largest \|logit\| in those batches | 19.81 |
| Relative difference | 7.2e-7 (float32 rounding) |
| Dense val LM loss | 1.81792206 |
| MoE val LM loss (exact clone) | 1.81792204 |
| MoE val LM loss after 1e-3 noise | 1.81800040 (+0.00008) |

Why it holds: all experts equal `F`, and the top-2 weights are renormalised so
`w_a + w_b = 1`, hence `y = w_a F(x) + w_b F(x) = F(x)` whatever the router picks.

Re-running the script reproduces these exact values.

### Q3. Does the MoE keep training and reducing loss?

Validation LM loss after checkpoint C:

| steps after C | Dense control | MoE, exact clone |
|---|---|---|
| 0 | 1.8179 | 1.8179 |
| 100 | 1.8383 | 1.8414 |
| 200 | 1.8335 | 1.8386 |
| 500 | 1.8053 | 1.7862 |
| 1,000 | 1.7415 | 1.6951 |
| 1,500 | 1.7105 | 1.6566 |

Yes: 1.8179 → 1.6566.

**The bump at steps 100–200 is not caused by conversion.** The dense control
shows it too. Phase A ended at LR 1e-4; both continuation branches re-warm to
5e-4 with a fresh optimizer, which briefly raises the loss. Without the control
this would have been mistaken for a conversion cost.

### Ablation: does tiny noise help?

Same router init, same data order, only difference = 1e-3 noise on expert weights
after the equivalence check.

| steps after C | MoE, exact clone | MoE, + 1e-3 noise |
|---|---|---|
| 0 | 1.8179 | 1.8180 |
| 500 | 1.7862 | 1.7907 |
| 1,000 | 1.6951 | 1.6974 |
| 1,500 | 1.6566 | 1.6589 |

| at 1,500 steps | exact clone | + noise |
|---|---|---|
| weight cosine | 0.9167 | 0.9165 |
| output cosine | 0.7911 | 0.7949 |
| load CV | 0.103 | 0.088 |

In this run the noise made no meaningful difference: by step 500 the two runs
have diverged to the same degree, and final losses differ by 0.002. Routing
alone breaks symmetry. One seed only, so read this as "no visible effect here",
not as a general result.


### Q4. Does it actually behave like an MoE?

MoE, exact clone (no noise), mean over the 4 MoE layers:

| steps after C | load CV | weight cosine | output cosine |
|---|---|---|---|
| 0 | 0.461 | 1.000 | 1.000 |
| 500 | 0.176 | 0.950 | 0.841 |
| 1,000 | 0.136 | 0.922 | 0.799 |
| 1,500 | 0.103 | 0.917 | 0.791 |

- **Load balance improves.** CV (std / mean of each expert's token share) falls
  from 0.461 to 0.103. Final shares are 10.4%–16.2% per expert (perfect = 12.5%).
- **Experts specialise without any noise.** Different tokens reach different
  experts from step 1, so their gradients differ. Output cosine falls from 1.000
  to 0.791, a larger drop than weight cosine (0.917), so the experts compute
  different functions, not just slightly different weights.

**A note on the aux loss.** It stayed between 3.9975 and 4.0221 (100-step window means) (perfect = 4.0 over
4 layers) the whole time, even at step 0 when load CV was 0.461. Because the
router is initialised small, its probabilities `P` are nearly uniform, and
`E * sum(f_i * P_i)` is close to 1 whenever *either* `f` or `P` is uniform. So
here the aux value is a weak imbalance meter; CV of the actual token counts is
the better diagnostic.

**Router gradients** (exact clone):

| measured at | \|\|grad router L_LM\|\| | \|\|grad router L_total\|\| |
|---|---|---|
| conversion (0 updates) | 8.45e-9 | 1.24e-2 |
| after 24 updates | 8.90e-3 | 1.05e-2 |
| end of training | 6.84e-2 | 6.84e-2 |

At conversion the LM loss gives the router essentially zero gradient (the output
does not depend on router weights), so the router first learns only from the aux
loss. Within 24 updates the experts have diverged enough for the LM loss to take
over.

---

## What this does NOT show

**It does not show MoE beats dense.** The MoE ends lower (1.6566 vs 1.7105), but
top-2 routing runs two FFNs per token: 1.87M active params vs 1.08M. That is not
an equal-compute comparison. The assignment asks only that the converted model
keeps training, which it does.

Other limits: one seed; a tiny model on a tiny dataset; char-level tokens.

---

## Install and run

Dependencies are `torch` and `matplotlib`. There is no `pyproject.toml`; `uv` installs them when you run a script.

Install [uv](https://docs.astral.sh/uv/) if you do not already have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

From this directory:

```bash
# smoke test, ~2 min, writes results_quick/
uv run --with torch --with matplotlib python run_all.py --quick

# full experiment, ~45 min on one CPU core, writes results/
# resumes if interrupted: finished stages are skipped
uv run --with torch --with matplotlib python run_all.py

# plots: results/plot*.png and results/summary.json
uv run --with torch --with matplotlib python plot_results.py
```

`uv run --with` builds a temporary environment and reuses it on later runs.

To keep a project virtualenv instead:

```bash
uv venv
source .venv/bin/activate
uv pip install torch matplotlib
python run_all.py --quick
python run_all.py
python plot_results.py
```

## Files

| File | Role |
|---|---|
| `data.py` | TinyShakespeare download, char tokenizer, seeded train batches, fixed val batches |
| `model.py` | GPT, `SwiGLU`, `DenseFFN`, `MoEFFN`, `count_params` |
| `convert_to_moe.py` | `convert_to_moe` (exact clone), `perturb_experts` (separate noise step) |
| `evaluate.py` | val loss, equivalence check, MoE diagnostics |
| `train.py` | one training loop for every phase, router-gradient measurement |
| `run_all.py` | runs the experiment tree |
| `plot_results.py` | three plots + summary |
| `WALKTHROUGH_hinglish.md` | beginner walkthrough of the code, in Hinglish |

## Plots

- `results/plot1_main_result.png` — validation LM loss, whole run + zoom
- `results/plot2_moe_objective.png` — train LM vs total loss, aux loss
- `results/plot3_moe_diagnostics.png` — load CV, final load heatmap, weight/output divergence, router gradients
