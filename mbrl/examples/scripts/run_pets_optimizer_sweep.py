#!/usr/bin/env python3
"""Run PETS optimizer sweeps over dynamics models and optimizers.

This is a thin convenience wrapper around `mbrl.examples.main` so you don't have to
remember the Hydra syntax. By default it uses the override expression defined in
`conf/override_expr/pets_optimizer_benchmark.yaml`.
"""

import argparse
import subprocess
import sys
from typing import Iterable, List

DEFAULT_DYNAMICS_MODELS = [
    "mlp",
    "mlp_ensemble",
    "gaussian_mlp",
    "gaussian_mlp_ensemble",
]

DEFAULT_OPTIMIZERS = [
    "cem",
    "decent_cem",
    "bccem",
    "gmm_cem",
    "icem",
    "nes",
    "cma_es",
    "mppi",
]


def parse_csv_list(value: str, fallback: List[str]) -> List[str]:
    if value is None:
        return fallback
    items = [x.strip() for x in value.split(",") if x.strip()]
    return items or fallback


def run_command(cmd: List[str], dry_run: bool) -> int:
    print(" ".join(cmd))
    if dry_run:
        return 0
    completed = subprocess.run(cmd)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep PETS over dynamics models and action optimizers."
    )
    parser.add_argument(
        "--override-expr",
        default="pets_optimizer_benchmark",
        help="Name of override expression config (under conf/override_expr).",
    )
    parser.add_argument(
        "--overrides",
        default=None,
        help="Optional overrides config (environment) to use instead of the one in the override expression.",
    )
    parser.add_argument(
        "--dynamics-models",
        default=",".join(DEFAULT_DYNAMICS_MODELS),
        help="Comma-separated dynamics models to run.",
    )
    parser.add_argument(
        "--optimizers",
        default=",".join(DEFAULT_OPTIMIZERS),
        help="Comma-separated action optimizers to run.",
    )
    parser.add_argument(
        "--seeds",
        default="0",
        help="Comma-separated seeds to run.",
    )
    parser.add_argument(
        "--experiment",
        default="pets_optimizer_benchmark",
        help="Value to set for experiment in Hydra config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the commands that would be executed.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failure instead of continuing.",
    )
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Additional Hydra overrides (prefix with -- to separate).",
    )

    args = parser.parse_args()

    dynamics_models = parse_csv_list(args.dynamics_models, DEFAULT_DYNAMICS_MODELS)
    optimizers = parse_csv_list(args.optimizers, DEFAULT_OPTIMIZERS)
    seeds = parse_csv_list(args.seeds, ["0"])

    failures: List[str] = []
    for dm in dynamics_models:
        for opt in optimizers:
            for seed in seeds:
                cmd = [
                    sys.executable,
                    "-m",
                    "mbrl.examples.main",
                    f"+override_expr={args.override_expr}",
                    f"dynamics_model={dm}",
                    f"action_optimizer={opt}",
                    f"seed={seed}",
                    f"experiment={args.experiment}",
                ]
                if args.overrides:
                    cmd.append(f"overrides={args.overrides}")
                if args.extra:
                    cmd.extend(args.extra)

                code = run_command(cmd, args.dry_run)
                if code != 0:
                    failures.append(" ".join(cmd))
                    if args.fail_fast:
                        print("Stopping because --fail-fast was set.")
                        return code

    if failures:
        print("\nFailed runs:")
        for cmd in failures:
            print(cmd)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
