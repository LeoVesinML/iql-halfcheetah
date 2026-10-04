# Team contribution workflow

`main` is the integrated baseline. @MedvAx-AI reviews interface changes and PRs.
See role ownership and acceptance criteria in [IQL_5_Person_Project_Plan.md](IQL_5_Person_Project_Plan.md).

## Work on your role branch

```bash
git clone https://github.com/MedvAx-AI/iql-halfcheetah.git
cd iql-halfcheetah
git switch feature/environment-dataset  # substitute your role branch
git fetch origin
git merge origin/main
uv sync --frozen --extra dev --extra notebook
```

Keep the four role branches available during the project. Push small commits,
open a draft PR to `main` early, reference the role issue, and request review from
@MedvAx-AI. Participants need repository write access, or may fork and propose a PR;
the lead grants collaborator access after receiving their actual GitHub usernames.

Before marking ready:

```bash
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen pytest
uv run --frozen python scripts/check_environment.py
uv run --frozen python scripts/check_notebook.py
```

Add meaningful tests for episode boundaries, masks, numerical loss cases,
preprocessing/checkpoint parity and evaluation behavior in the owning PR.
Do not weaken scaffold tests to accept broken interfaces. Update all consumers
when a signature changes; discuss shared config/contracts with Person 1 first.

## Merge rules

- All changes use PRs; successful CI and lead review are the acceptance gates.
- Prefer squash merge with a concrete behavior description.
- After a merge, fetch and merge `origin/main` into every active role branch.
- Never force-push another participant's work. Resolve shared-file conflicts with
  the lead. Keep notebooks output-free during drafting.
- Do not automatically delete role branches after merges; use fresh task branches
  off updated `main` if a role needs additional independent PRs.

CODEOWNERS records the lead, but enforcement depends on repository settings and
GitHub plan. See `docs/VERIFICATION.md` for the actual protection state.

## Dependency changes

Edit `pyproject.toml`, run `uv lock`, sync and test, then export requirements with
the README command. Commit pyproject, uv.lock and requirements.lock.txt together.
Never hand-edit the export or claim
a CUDA setup tested when only the CPU profile passed.

## Artifacts and provenance

Generated datasets, checkpoints, logs and videos are ignored. Small reviewed result
summaries may be deliberately added with `git add -f results/<file>`; use Releases
or a documented team storage URL for large artifacts and record SHA-256 checksums.
Every result must identify source commit, run ID, seed, dataset, config, versions,
hardware, transforms and checkpoint. No credentials or private machine paths.

## Definition of ready for final integration

Role issue checkboxes are satisfied, tests pass, examples use public interfaces,
limitations are stated, and artifacts can be retrieved. Person 1 then checks the
full pipeline and clean Colab. Learning performance is assessed from real raw
returns and the demonstration, never from installation or CI success alone.
