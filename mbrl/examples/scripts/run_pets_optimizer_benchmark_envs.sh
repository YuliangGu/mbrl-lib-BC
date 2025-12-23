#!/usr/bin/env bash
set -euo pipefail

# Shell wrapper for `mbrl.examples.scripts.run_pets_optimizer_benchmark`.
#
# Usage:
#   mbrl/examples/scripts/run_pets_optimizer_benchmark_envs.sh [pets_cartpole pets_inv_pendulum ...]

# Examples:
#   DEVICE=cuda:0 SEEDS="0 1" BUDGET_MODE=env_round \
#     mbrl/examples/scripts/run_pets_optimizer_benchmark_envs.sh pets_cartpole pets_inv_pendulum
#
#   OPTIMIZERS="bccem cem" EXTRA="overrides.num_steps=2000" \
#     mbrl/examples/scripts/run_pets_optimizer_benchmark_envs.sh pets_halfcheetah

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
SEEDS_STR="${SEEDS:-0}"
# OPTIMIZERS_STR="${OPTIMIZERS:-cem decent_cem bccem gmm_cem icem nes cma_es mppi}"
OPTIMIZERS_STR="${OPTIMIZERS:-cem bccem}"
BUDGET_MODE="${BUDGET_MODE:-fixed}"
OVERRIDE_EXPR="${OVERRIDE_EXPR:-pets_optimizer_benchmark}"
OUTPUT_ROOT="${OUTPUT_ROOT:-exp/pets_optimizer_benchmark_runs}"
EXTRA_STR="${EXTRA:-}"
EXTRA_BCCEM_STR="${EXTRA_BCCEM:-}"
DEBUG="${DEBUG:-0}"
DRY_RUN="${DRY_RUN:-0}"

read -r -a SEEDS_ARR <<< "${SEEDS_STR}"
read -r -a OPTIMIZERS_ARR <<< "${OPTIMIZERS_STR}"
read -r -a EXTRA_ARR <<< "${EXTRA_STR}"
read -r -a EXTRA_BCCEM_ARR <<< "${EXTRA_BCCEM_STR}"

if (($# > 0)); then
  ENVS_ARR=("$@")
else
  ENVS_ARR=(pets_cartpole pets_inv_pendulum pets_hopper pets_halfcheetah)
fi

CMD=(
  "${PYTHON_BIN}" -m mbrl.examples.scripts.run_pets_optimizer_benchmark
  --override-expr "${OVERRIDE_EXPR}"
  --envs "${ENVS_ARR[@]}"
  --optimizers "${OPTIMIZERS_ARR[@]}"
  --seeds "${SEEDS_ARR[@]}"
  --budget-mode "${BUDGET_MODE}"
  --output-root "${OUTPUT_ROOT}"
  --extra "device=${DEVICE}" "${EXTRA_ARR[@]}"
)

if ((${#EXTRA_BCCEM_ARR[@]} > 0)); then
  CMD+=(--extra-bccem "${EXTRA_BCCEM_ARR[@]}")
fi
if [[ "${DEBUG}" == "1" ]]; then
  CMD+=(--debug)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  CMD+=(--dry-run)
fi

exec "${CMD[@]}"
