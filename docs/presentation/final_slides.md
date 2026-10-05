# Final presentation slides — Person 5

Fill measured numbers and the video URL after Persons 3–4 finish. Do not invent returns.

---

## Slide 1 — Problem

Offline RL on HalfCheetah: learn from a fixed dataset only, then control the cheetah online. Challenge: avoid overestimating actions missing from the data.

---

## Slide 2 — Environment

HalfCheetah-v5 · obs `(17,)` · action `(6,)` · forward reward − control cost · recovered EnvSpec must match the Minari dataset.

---

## Slide 3 — Dataset

`mujoco/halfcheetah/medium-v0` · 1M steps / 1k episodes (per dataset card) · float32 training contract · separate `terminated` vs `truncated` flags.

---

## Slide 4 — IQL (three losses)

| Loss | Role |
|---|---|
| Expectile V | Fit V to upper expectile of in-dataset Q |
| Masked Q | `y = r + γ (1−terminated) V(s')` |
| Capped AWR policy | `w = min(exp(β A), w_max)`, β = **inverse temperature** |

Config defaults: `expectile=0.7`, `inverse_temperature=3.0`, `max_weight=100`.

---

## Slide 5 — Manual numerical example

Given `Q=8`, `V=5`, `V'=6`, `r=1`, `γ=0.99`, `τ=0.7`, `β=3`:

- Expectile weight on `u=3`: `0.7` → V term `0.7 × 9 = 6.3`
- Q target (not terminated): `1 + 0.99×6 = 6.94` (terminated → `1.0`)
- Policy weight: `exp(9) ≈ 8103` → **capped to 100**

Same formulas as in `notebooks/iql_halfcheetah.ipynb` §6.

---

## Slide 6 — Experiment protocol

- Config: `configs/halfcheetah.toml`
- Train seeds / eval seeds recorded in `manifest.json`
- Metrics: raw returns in `evaluation.csv` + `summary.json`
- Hardware / wall-clock: _TBD from Person 3–4 run_

---

## Slide 7 — Measured results

| Policy | Mean return | Std | Seeds |
|---|---|---|---|
| Random baseline | _TBD_ | _TBD_ | _TBD_ |
| IQL agent | _TBD_ | _TBD_ | _TBD_ |

Plot: `_TBD path under results/<run_id>/plots/_`

---

## Slide 8 — Limitations + demo + references

**Limitations:** offline coverage of medium data; hyperparameter sensitivity; render/GL on Colab; no invented scores before runs land.

**Demo video:** _link MP4 here_ (`videos/<run_id>/...`)

**References:** Kostrikov et al. ICLR 2022; Minari medium-v0; Gymnasium HalfCheetah; project `docs/INTERFACES.md`.
