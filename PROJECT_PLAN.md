# IQL HalfCheetah — five-person project plan

**Goal:** one reproducible repository and Colab explaining IQL and demonstrating
an agent trained on a fixed offline dataset in HalfCheetah-v5.

**Architecture:** [Integration design](docs/ARCHITECTURE.md). One Python 3.11 package
with shared NumPy batch contracts; PyTorch implements the learner. Gymnasium and
Minari provide environment and dataset. Each role owns an isolated feature branch.

## Person 1 setup implementation plan

- [x] Create packaging, exact direct dependency pins, lockfile, configuration and
  artifact directories. Verify an isolated install with `pip check` and wheel build.
- [x] Add contract tests before implementing config parsing and scaffold CLI:
  valid defaults, invalid numeric/type/unknown fields, boolean discount, CLI success,
  unfinished entry points exit nonzero without producing artifacts.
- [x] Define dataset/environment/learner/train/evaluate interfaces and actionable
  stubs. Verify imports and tests without downloading data or training a model.
- [x] Add contribution guide, CI, templates and valid unexecuted notebook outline.
  Verify notebook schema and real HalfCheetah reset/step in the pinned environment.
- [x] Publish to MedvAx-AI, create four feature branches and four measurable role
  issues. Read remote state and CI; record evidence and handoff links.

## Ownership

| Person | Branch | Files | Deliverable |
|---|---|---|---|
| 1 | main (integration through PRs) | packaging/config/contracts/CI/docs | installable scaffold; later integrated release and clean Colab check |
| 2 | feature/environment-dataset | dataset.py, environment.py, data documentation | environment factory, loader, normalization, deterministic sampler |
| 3 | feature/iql | networks.py, iql.py, train.py | tested IQL update, training orchestration, checkpoint/resume |
| 4 | feature/evaluation | evaluate.py, results/, videos/ | evaluation protocol, curves, baseline and agent video |
| 5 | feature/notebook-presentation | notebooks/, docs/presentation/ | runnable narrative notebook, numerical example, proposal/final slides |

GitHub usernames for Persons 2–5 are not supplied. Issues use role ownership and
remain unassigned until participants claim them; no access invitations are assumed.

## Roadmap and gates

1. **M0 — scaffold:** install, contracts, CI and four role issues pass. Person 1.
2. **M1 — proposal and data contract:** Person 2 confirms exact dataset metadata,
   recovery environment and preprocessing; Person 3 documents loss conventions;
   Person 5 prepares 5–6 proposal slides. Person 1 reviews compatibility.
3. **M2 — minimal integration:** dataset yields one batch; IQL performs one finite
   update; evaluator completes one episode; normalization survives checkpoint.
   Person 1 merges reviewed PRs and checks joint smoke run.
4. **M3 — experiments:** Persons 3–4 run at least seeds 0, 1, 2 on fixed offline
   data, evaluate at agreed checkpoints on separate evaluation seeds, save manifest
   and raw metrics. Proposed full budget is 500,000 updates, subject to measured
   hardware throughput; start with a short 1,000-update sanity run.
5. **M4 — final release:** Person 4 supplies return statistics, plots and MP4;
   Person 5 completes numerical example and slides; Person 1 tests a fresh Colab
   Run all, verifies artifact links/checksums, tags a release and freezes versions.

These are dependency gates, not invented deadlines. Completion dates are set by
the team against the course deadline and measured training cost.

## Definition of done per role

### Person 2 — Environment & Dataset

- Load the configured Minari dataset; download only on explicit request. Record
  ID, episode/transition counts, source license and recovered EnvSpec.
- Validate continuous observation `(17,)` and action `(6,)` spaces and bounds.
- Flatten each episode using observations `[:-1]` and `[1:]`; preserve float32
  numeric data and separate bool flags. Prove no cross-episode next-state links.
- Deterministic seeded sampling returns the batch contract. Test episode length,
  dtype, shapes, NaN rejection and last-transition truncation.
- Compute normalization from offline data only; document reward transform and
  save statistics for identical evaluation preprocessing.
- Demonstrate recovered environment reset and at least 10 steps; document reward,
  goal and horizon. PR includes small fixtures, no full dataset in Git.

### Person 3 — IQL Implementation & Training

- Implement twin Q, V and bounded continuous policy, upper-expectile V loss,
  Bellman Q target and advantage-weighted behavior cloning. Document target-Q
  choice, stop gradients, target soft-update and action likelihood convention.
- Use `target = reward + discount * (1 - terminated) * V(next_observation)`;
  preserve time-limit bootstrapping and end evaluation on either flag.
- Policy uses `exp(inverse_temperature * advantage)` capped by max weight;
  log finite q_loss, v_loss, policy_loss and advantage_mean.
- Test analytical loss cases, masks, finite updates, action bounds and gradient
  isolation. Run at least 1,000 updates on real data without divergence.
- Training accepts shared config, logs update-indexed JSONL, seeds RNGs, and
  saves networks/optimizers/step/RNG/config/data ID/preprocessing/version metadata.
  Prove save/load action parity and resume step progression on a tiny fixture.

### Person 4 — Experiments & Evaluation

- Evaluate deterministic policy on the recovered environment with checkpoint
  normalization; close environments and never add evaluation data to training.
- At least 10 episodes for each of 3 training seeds; retain each raw return,
  episode length and evaluation seed. Report within-run and across-run statistics
  separately. Reuse fixed evaluation seeds across checkpoints and baseline.
- Include random-policy baseline under identical settings; any extra baseline is
  optional. Report raw episodic return; normalized scores require sourced reference
  returns and identical environment/reward settings. Do not assume D4RL equivalence.
- Publish CSV/JSON metrics, training/evaluation curves and a playable MP4 in the
  agreed artifact location with run ID and metadata. No invented improvement goal.
- Tests cover seed repeatability, both ending flags, preprocessing, return sum and
  video path; document hardware, elapsed time and any failed runs.

### Person 5 — Theory, Notebook & Presentation

- Replace the supplied 11-section notebook outline with narrative plus imports
  from the package. No duplicated learner implementation in notebook cells.
- Explain offline learning and all three losses; work one numerical transition
  including expectile weight, masked Q target and capped policy weight. State
  whether beta is temperature or inverse temperature and use our convention.
- Proposal: 5–6 slides. Final: problem, environment, dataset, IQL, manual example,
  experiment protocol, measured results, limitations and linked video.
- Cite paper, dataset and any adapted source. Clear secrets and large outputs;
  execute notebook from a clean Colab after M2–M3 and record tested commit/runtime.

## Integration and communication

Each member opens a draft PR early and updates their issue with blockers and
artifact links. Interface changes need Person 1 review and updates to consumers.
Merge order: data contract → learner/training → evaluation → final notebook.
Persons 4–5 can prepare protocols/narrative in parallel before results arrive.

## Risks and responses

| Risk | Response / owner |
|---|---|
| v4/D4RL data mixed with v5 evaluation | recover exact Minari EnvSpec; Person 2 |
| time limits treated as terminal | separate flags and mask tests; Persons 2–3 |
| normalization differs at evaluation | store stats in checkpoint; Persons 2–4 |
| training too slow / unstable | measure short run, revise budget transparently; Persons 3–4 |
| rendering fails in Colab | separate numerical smoke and render check; Person 4 |
| notebook depends on local state | fresh Run all at fixed commit; Persons 1 & 5 |

## Final defense

Person 1: problem and pipeline. Person 2: environment/data. Person 3: algorithm.
Person 4: experiments/results. Person 5: numerical example and demo/conclusion.

## Current status

M0 is the deliverable of this setup; see [handoff links](docs/TEAM_HANDOFF.md).
M1–M4 remain team work. Installation and
contract checks do not establish learning quality or a completed final Colab.
