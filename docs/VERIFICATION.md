# Scaffold verification and handoff

Checked on **2026-10-04**. This report concerns Person 1's scaffold; it does not
claim an implemented algorithm, trained checkpoint, results or final Colab.

## Local checks

Reference local runtime: Windows x64, Python 3.11.15, isolated `.venv` created
from `uv.lock` using uv 0.8.22 with dev/notebook extras and CPU PyTorch 2.7.1.

- Dependency consistency: 155 installed packages compatible (`uv pip check`).
- Contract suite: 14 tests passed, including invalid config and explicit unfinished
  train/evaluate exits with no artifacts. First run failed before package creation.
- Ruff lint and format checks: passed.
- Real Gymnasium HalfCheetah-v5: reset and ten steps, finite observations/rewards.
- Notebook: nbformat schema valid; code outputs empty and execution counts cleared.

The pinned numerical environment works locally. No dataset is downloaded and no
training or rendering is performed. Linux checks run in GitHub Actions after publication.
Build, remote CI and protection evidence will be recorded below after completion.

## Final team checks still required

Dataset metadata and episode-boundary validation (Person 2); IQL numerical correctness,
real-data updates, checkpoint/resume and GPU profile if used (Person 3); actual
seeded returns and headless video (Person 4); completed narrative/manual example/slides
(Person 5); integrated fresh Colab Run all and final release (Person 1).

Participant usernames were not provided. Role issues are intentionally unassigned;
the lead assigns them and grants access when those usernames are available.
