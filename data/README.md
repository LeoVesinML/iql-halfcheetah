# Offline data

Store generated/downloaded data outside Git. Default Minari cache is its standard
user cache; optionally set `MINARI_DATASETS_PATH` to an absolute local `data/minari`
directory. Keep the selected dataset ID in config and manifest.

Person 2 owns the loader. The explicit download command after installation is:

```bash
minari download mujoco/halfcheetah/medium-v0
```

The scaffold does not execute it. Record source/license/metadata and preserve
episode boundaries. Do not mix D4RL/v4 data into the v5 contract.
