"""Person 2 acceptance: full data validation, seeded batch and recovered-env smoke."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, fields
from pathlib import Path

import minari
import numpy as np

from iql_project.config import load_config
from iql_project.dataset import load_offline_dataset
from iql_project.environment import make_evaluation_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/halfcheetah.toml")
    parser.add_argument(
        "--download", action="store_true", help="Explicitly allow a missing download"
    )
    parser.add_argument("--datasets-path", type=Path, help="Optional local Minari cache directory")
    parser.add_argument("--output", type=Path, default=Path("results/person-2/dataset_report.json"))
    args = parser.parse_args()
    if args.datasets_path is not None:
        os.environ["MINARI_DATASETS_PATH"] = str(args.datasets_path.resolve())
    config = load_config(args.config)
    start = time.perf_counter()
    dataset = load_offline_dataset(config, download=args.download)
    batch = dataset.sample(config.batch_size, np.random.default_rng(config.seed))
    repeated = dataset.sample(config.batch_size, np.random.default_rng(config.seed))
    for field in fields(batch):
        np.testing.assert_array_equal(getattr(batch, field.name), getattr(repeated, field.name))
    env = make_evaluation_env(config)
    try:
        observation, _ = env.reset(seed=config.seed)
        assert observation.shape == (config.observation_dim,)
        assert np.isfinite(observation).all()
        env.action_space.seed(config.seed)
        for step in range(10):
            observation, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            assert observation.shape == (config.observation_dim,)
            assert np.isfinite(observation).all() and np.isfinite(reward)
            if terminated or truncated:
                env.reset(seed=config.seed + step + 1)
        recovered_env_spec = json.loads(env.spec.to_json())
    finally:
        env.close()
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    root = Path(__file__).resolve().parents[1]
    source_files = [
        root / "src/iql_project/dataset.py",
        root / "src/iql_project/environment.py",
        Path(__file__),
        Path(args.config).resolve(),
    ]
    data_path = minari.load_dataset(config.dataset_id, download=False).storage.data_path
    data_checksums = {}
    for path in sorted(data_path.glob("*.hdf5")):
        with path.open("rb") as stream:
            data_checksums[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        **dataset.metadata,
        "config": asdict(config),
        "source_commit": git.stdout.strip() if git.returncode == 0 else None,
        "source_has_local_changes": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "source_files_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files
        },
        "data_files_sha256": data_checksums,
        "seed": config.seed,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "minari", "gymnasium", "mujoco", "torch")
        },
        "python_version": platform.python_version(),
        "recovered_env_spec": recovered_env_spec,
        "checks": {"all_transitions_validated": True, "seeded_sampling": True, "env_steps": 10},
        "elapsed_seconds": time.perf_counter() - start,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    checksum = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps(report["statistics"], indent=2))
    print(f"Validated {len(dataset):,} transitions; recovered environment reset + 10 steps passed")
    print(f"Report: {args.output}; SHA-256: {checksum}")


if __name__ == "__main__":
    main()
