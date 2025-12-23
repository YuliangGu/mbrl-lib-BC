#!/usr/bin/env bash
set -euo pipefail

# Ablation runner: vary dynamics ensemble size + #elites and run the PETS optimizer benchmark.
#
# Usage:
#   mbrl/examples/scripts/run_pets_ensemble_size_ablation.sh [pets_cartpole pets_inv_pendulum ...]
#
# Examples:
#   DEVICE=cuda:0 SEEDS="0 1" ENSEMBLE_SIZES="1 3 5 7" \
#     mbrl/examples/scripts/run_pets_ensemble_size_ablation.sh pets_cartpole
#
#   # Provide explicit elites (same length as ENSEMBLE_SIZES).
#   ENSEMBLE_SIZES="3 5 7" ELITES="1 3 5" OPTIMIZERS="cem bccem" \
#     mbrl/examples/scripts/run_pets_ensemble_size_ablation.sh pets_cartpole pets_inv_pendulum

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
SEEDS_STR="${SEEDS:-0}"
OPTIMIZERS_STR="${OPTIMIZERS:-cem decent_cem bccem gmm_cem icem nes cma_es mppi}"
BUDGET_MODE="${BUDGET_MODE:-fixed}"
OVERRIDE_EXPR="${OVERRIDE_EXPR:-pets_optimizer_benchmark_ensemble_size}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-exp/pets_ensemble_size_ablation_runs}"
EXTRA_STR="${EXTRA:-}"
EXTRA_BCCEM_STR="${EXTRA_BCCEM:-}"
DEBUG="${DEBUG:-0}"
DRY_RUN="${DRY_RUN:-0}"

# Defaults: match current baseline (7 members, 5 elites) + reduced ensembles.
ENSEMBLE_SIZES_STR="${ENSEMBLE_SIZES:-3 5 7}"
ELITES_STR="${ELITES:-}"

read -r -a SEEDS_ARR <<< "${SEEDS_STR}"
read -r -a OPTIMIZERS_ARR <<< "${OPTIMIZERS_STR}"
read -r -a EXTRA_ARR <<< "${EXTRA_STR}"
read -r -a EXTRA_BCCEM_ARR <<< "${EXTRA_BCCEM_STR}"
read -r -a ENSEMBLE_SIZES_ARR <<< "${ENSEMBLE_SIZES_STR}"
read -r -a ELITES_ARR <<< "${ELITES_STR}"

if (($# > 0)); then
  ENVS_ARR=("$@")
else
  ENVS_ARR=(pets_cartpole pets_inv_pendulum)
fi

if ((${#ELITES_ARR[@]} > 0)) && ((${#ELITES_ARR[@]} != ${#ENSEMBLE_SIZES_ARR[@]})); then
  echo "ELITES must be empty or have the same number of entries as ENSEMBLE_SIZES." >&2
  echo "ENSEMBLE_SIZES=${ENSEMBLE_SIZES_STR}" >&2
  echo "ELITES=${ELITES_STR}" >&2
  exit 2
fi

for i in "${!ENSEMBLE_SIZES_ARR[@]}"; do
  ens="${ENSEMBLE_SIZES_ARR[$i]}"
  if ((${#ELITES_ARR[@]} > 0)); then
    elites="${ELITES_ARR[$i]}"
  else
    # Default mapping: preserve the common "7 members / 5 elites" pattern => elites = max(1, ens - 2).
    elites=$((ens - 2))
    if ((elites < 1)); then elites=1; fi
    if ((elites > ens)); then elites="${ens}"; fi
  fi

  out_root="${OUTPUT_ROOT_BASE}/ens${ens}_elite${elites}"
  echo "=== ensemble_size=${ens} num_elites=${elites} output_root=${out_root} ==="

  CMD=(
    "${PYTHON_BIN}" -m mbrl.examples.scripts.run_pets_optimizer_benchmark
    --override-expr "${OVERRIDE_EXPR}"
    --envs "${ENVS_ARR[@]}"
    --optimizers "${OPTIMIZERS_ARR[@]}"
    --seeds "${SEEDS_ARR[@]}"
    --budget-mode "${BUDGET_MODE}"
    --output-root "${out_root}"
    --extra
      "device=${DEVICE}"
      "overrides.dyn_ensemble_size=${ens}"
      "overrides.dyn_num_elites=${elites}"
      "${EXTRA_ARR[@]}"
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

  "${CMD[@]}"
done

