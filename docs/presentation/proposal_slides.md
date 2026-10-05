# Proposal slides — Person 5 (5–6 slides)

Use this as speaker notes / paste into Google Slides / PowerPoint. Keep one slide per section.

---

## Slide 1 — Title / problem

**Implicit Q-Learning on HalfCheetah-v5**  
Offline RL course project · 5-person team · lead MedvAx-AI

**Problem.** Learn a strong continuous-control policy from a **fixed** offline dataset, then run it in Gymnasium HalfCheetah-v5 — without collecting new environment interactions during training.

---

## Slide 2 — Environment

**HalfCheetah-v5** (Gymnasium / MuJoCo)

- Observation: `(17,)` continuous body/joint state
- Action: `(6,)` continuous torques in `[-1, 1]`
- Goal: run forward efficiently
- Reward: forward progress − control cost

Why it fits IQL: continuous actions, standard offline locomotion benchmark, dataset and eval env share the same spec when recovered via Minari.

---

## Slide 3 — Offline dataset

**Minari:** `mujoco/halfcheetah/medium-v0` (from HalfCheetah-v5)

- Transitions: `(s, a, r, s', terminated, truncated)`
- Medium behavior policy — neither random nor expert
- Training uses **only** this fixed buffer (Person 2: load, normalize, sample)

Key contract: termination masks Q bootstrap; truncation does **not**.

---

## Slide 4 — IQL idea

Standard offline Q-learning can overestimate **out-of-distribution** actions (`max` over unseen `a'`).

**IQL** (Kostrikov et al., ICLR 2022):

1. **V** — upper expectile regression on **dataset** actions  
2. **Q** — backup with `V(s')`, no max over actions  
3. **π** — advantage-weighted regression on dataset actions  

Project convention: **β = inverse temperature** (`inverse_temperature=3.0`), weights capped by `max_weight=100`.

---

## Slide 5 — Pipeline / team plan

```text
Dataset (P2) → IQL learner (P3) → Train → Eval + video (P4)
                      ↓
              Notebook + slides (P5) → Colab acceptance (P1)
```

| Person | Deliverable |
|---|---|
| 1 | Repo, interfaces, final Colab |
| 2 | Env + Minari loader |
| 3 | Networks, losses, train |
| 4 | Metrics, plots, MP4 |
| 5 | Theory, manual example, slides |

---

## Slide 6 — Intended evaluation / demo

- Train offline for configured steps (`configs/halfcheetah.toml`)
- Evaluate raw episode returns vs random baseline (multi-seed)
- Record HalfCheetah MP4 demo
- Notebook: offline RL explanation + **one worked numerical transition**
- Fresh Google Colab `Run all` at a recorded commit

*(Measured numbers and video link come in the final slides after M2–M3.)*
