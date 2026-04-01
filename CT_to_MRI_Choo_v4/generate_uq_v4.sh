#!/bin/bash
# =============================================================================
# generate_uq_v4.sh
# UQ generation dispatcher for CT-to-MRI v4.
#
# Supports two UQ modes:
#
#   MODE=ensemble  (default)
#     Calls generate_uq_ensemble_v4.py which:
#       - Loads each of N=5 checkpoints SEQUENTIALLY on a single GPU
#       - Generates all test subjects per checkpoint
#       - Clears GPU memory between checkpoints (torch.cuda.empty_cache)
#       - Computes pixel-wise mean and variance across the ensemble
#
#   MODE=mc_dropout
#     Calls generate_bbdm_ct2mri_v4.py --mc_dropout which:
#       - Loads ONE checkpoint
#       - Runs num_mc_samples forward passes (MCDropout active in eval mode)
#       - Saves mean and variance per subject
#
# Usage:
#   bash generate_uq_v4.sh             # ensemble mode (default)
#   MODE=mc_dropout bash generate_uq_v4.sh
# =============================================================================

set -e

# ---- User configuration (edit these) ----------------------------------------
CKPT_ROOT="/pscratch/sd/s/seojw/CT_to_MRI/checkpoints"
BASE_EXP_NAME="bbdm_ct2mri_SynthRAD+severance_mni_n226_v4_ensemble"
CHECKPOINT_VER="epoch065"   # used for mc_dropout mode; ignored for ensemble mode

TEST_METADATA="/pscratch/sd/s/seojw/CT_to_MRI/metadata/test_metadata_total_n226_20260302.csv"
TRAIN_METADATA="/pscratch/sd/s/seojw/CT_to_MRI/metadata/train_metadata_total_n226_20260302.csv"

# Deep Ensemble: 5 checkpoint paths (one per member, in the same order as training)
SEEDS=(1337 42 123 456 789)

# MC Dropout settings (used only when MODE=mc_dropout)
NUM_MC_SAMPLES=10

# ISTA settings (applied in both modes)
USE_ISTA=true
NUM_ISTA_STEP=1
ISTA_STEP_SIZE=0.5
SUB_BATCH_SIZE=6

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Environment ------------------------------------------------------------
module load python 2>/dev/null || true
conda activate flashattention 2>/dev/null || true

# ---- Mode selection ---------------------------------------------------------
MODE="${MODE:-ensemble}"

if [[ "${MODE}" == "ensemble" ]]; then
    echo "============================================================"
    echo "Mode: Deep Ensemble (N=${#SEEDS[@]}, sequential GPU loading)"
    echo "============================================================"

    # Build the list of checkpoint paths
    CKPT_PATHS=()
    for SEED in "${SEEDS[@]}"; do
        EXP_NAME="${BASE_EXP_NAME}_seed${SEED}"
        CKPT_PATH="${CKPT_ROOT}/${EXP_NAME}/${EXP_NAME}_${CHECKPOINT_VER}.pt"
        if [[ ! -f "${CKPT_PATH}" ]]; then
            echo "ERROR: Checkpoint not found: ${CKPT_PATH}"
            exit 1
        fi
        CKPT_PATHS+=("${CKPT_PATH}")
    done

    OUTPUT_DIR="${CKPT_ROOT}/${BASE_EXP_NAME}_ensemble_uq/${CHECKPOINT_VER}"

    # Construct --ensemble_ckpt_paths args
    CKPT_ARGS=""
    for P in "${CKPT_PATHS[@]}"; do
        CKPT_ARGS+=" ${P}"
    done

    ISTA_FLAGS=""
    if [[ "${USE_ISTA}" == "true" ]]; then
        ISTA_FLAGS="--use_ista --num_ISTA_step ${NUM_ISTA_STEP} --ISTA_step_size ${ISTA_STEP_SIZE} --sub_batch_size ${SUB_BATCH_SIZE}"
    fi

    python "${SCRIPT_DIR}/generate_uq_ensemble_v4.py" \
        --ensemble_ckpt_paths ${CKPT_ARGS} \
        --test_metadata  "${TEST_METADATA}"  \
        --train_metadata "${TRAIN_METADATA}" \
        --output_dir     "${OUTPUT_DIR}"     \
        --use_bf16 \
        --mni \
        ${ISTA_FLAGS}

    echo ""
    echo "Ensemble UQ outputs saved to: ${OUTPUT_DIR}"

elif [[ "${MODE}" == "mc_dropout" ]]; then
    echo "============================================================"
    echo "Mode: MC Dropout (single checkpoint, ${NUM_MC_SAMPLES} passes)"
    echo "============================================================"

    # Use the first seed's checkpoint by default; override by setting MC_CKPT_PATH
    SEED="${SEEDS[0]}"
    EXP_NAME="${BASE_EXP_NAME}_seed${SEED}"
    MC_CKPT_PATH="${MC_CKPT_PATH:-${CKPT_ROOT}/${EXP_NAME}/${EXP_NAME}_${CHECKPOINT_VER}.pt}"

    if [[ ! -f "${MC_CKPT_PATH}" ]]; then
        echo "ERROR: Checkpoint not found: ${MC_CKPT_PATH}"
        exit 1
    fi

    ISTA_FLAGS=""
    if [[ "${USE_ISTA}" == "true" ]]; then
        ISTA_FLAGS="--use_ista --num_ISTA_step ${NUM_ISTA_STEP} --ISTA_step_size ${ISTA_STEP_SIZE} --sub_batch_size ${SUB_BATCH_SIZE}"
    fi

    python "${SCRIPT_DIR}/generate_bbdm_ct2mri_v4.py" \
        --experiment_name "${EXP_NAME}"       \
        --checkpoint_ver  "${CHECKPOINT_VER}" \
        --test_metadata   "${TEST_METADATA}"  \
        --train_metadata  "${TRAIN_METADATA}" \
        --mc_dropout \
        --num_mc_samples  "${NUM_MC_SAMPLES}" \
        --use_bf16 \
        --mni \
        ${ISTA_FLAGS}

    echo ""
    echo "MC Dropout UQ outputs saved to checkpoint syn/ directory."

else
    echo "ERROR: Unknown MODE='${MODE}'. Set MODE=ensemble or MODE=mc_dropout."
    exit 1
fi

echo "Done."
