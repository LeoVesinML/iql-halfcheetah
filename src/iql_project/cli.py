"""A scaffold diagnostic; no unfinished command reports simulated success."""

import argparse
import json
import sys
from dataclasses import asdict

from .config import ProjectConfig, load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IQL team project scaffold")
    parser.add_argument("command", choices=("check", "train", "evaluate"))
    parser.add_argument("--config", help="Flat TOML config; omit for built-in defaults")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config) if args.config else ProjectConfig()
        if args.command != "check":
            owner = "Person 3" if args.command == "train" else "Person 4"
            raise NotImplementedError(f"{owner}: {args.command} is not implemented")
        print(
            json.dumps(
                {
                    "status": "scaffold configuration valid; algorithm not implemented",
                    "config": asdict(config),
                    "owners": {
                        "dataset/environment": "Person 2",
                        "IQL/train": "Person 3",
                        "evaluation/video": "Person 4",
                        "notebook/slides": "Person 5",
                    },
                },
                indent=2,
            )
        )
        return 0
    except (OSError, ValueError, NotImplementedError) as error:
        print(str(error), file=sys.stderr)
        return 2
