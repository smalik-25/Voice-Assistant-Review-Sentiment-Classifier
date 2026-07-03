"""Config-driven experiment entrypoint.

Reads a YAML config, seeds every RNG from it, runs the variant named in the config,
and logs params and metrics to MLflow. At Phase 0 the only variant is ``noop``: it
draws one seeded random number and logs it as ``dummy_metric``. Two runs with the same
seed produce the same number, which is the reproducibility check for the harness.

Deliberately minimal. Config schemas and model abstractions are not built here; they
arrive when a real variant needs them (Phase 2 onward).

Usage:
    python -m src.run_experiment --config configs/noop.yaml
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

# The plan uses a local file-store MLflow backend (gitignored ``mlruns/``). MLflow 3.x
# gates the file store behind an opt-out, so enable it here before any store access.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import numpy as np
import yaml


def set_seeds(seed: int) -> None:
    """Seed the standard-library and numpy RNGs from the config.

    Framework seeds (torch, tensorflow) are set by their own variants when those
    land in Phase 3, from this same config value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def run_noop(config: dict) -> dict:
    """Stand-in experiment: a single seeded draw.

    Identical seeds give an identical value, which is exactly what the smoke test
    checks.
    """
    return {"dummy_metric": float(np.random.rand())}


def _run_sklearn_linear(config: dict) -> dict:
    # Imported lazily so a noop run does not pull in the modeling stack.
    from src.models.sklearn_linear import run

    return run(config)


VARIANTS = {"noop": run_noop, "sklearn_linear": _run_sklearn_linear}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a YAML experiment config")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    seed = int(config["seed"])
    variant = config["variant"]
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; known: {sorted(VARIANTS)}")

    set_seeds(seed)

    mlflow.set_tracking_uri(config.get("tracking_uri", "file:./mlruns"))
    mlflow.set_experiment(config.get("experiment", "default"))
    with mlflow.start_run(run_name=config.get("name", variant)):
        mlflow.log_param("seed", seed)
        mlflow.log_param("variant", variant)
        mlflow.log_param("config_path", str(args.config))
        metrics = VARIANTS[variant](config)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

    print(f"variant={variant} seed={seed} metrics={metrics}")


if __name__ == "__main__":
    main()
