# Offline data

Store generated/downloaded data outside Git. Default Minari cache is its standard
user cache; optionally set `MINARI_DATASETS_PATH` to an absolute local `data/minari`
directory. Keep the selected dataset ID in config and manifest.

Person 2's loader is implemented. The explicit download command after installation is:

```bash
minari download mujoco/halfcheetah/medium-v0
```

The loader defaults to `download=False`; opt in explicitly to download missing data.
To download into this ignored directory, validate every transition, save statistics
and run the recovered environment for ten steps:

```bash
uv run --frozen python scripts/check_dataset.py --download --datasets-path data/minari
```

Omit `--download` for subsequent local checks. To use that same cache in training
or a notebook, set `MINARI_DATASETS_PATH` in its process too. Generated reports go
to `results/person-2/`; `--output` overrides the report location.

See [the Person 2 handoff](../docs/ENVIRONMENT_DATASET.md) for measured statistics,
normalization, source/license metadata and the MuJoCo version limitation. Episode
boundaries and separate termination/truncation flags are preserved. Do not mix
D4RL/v4 data into the v5 contract.
