#!/bin/bash
# =============================================================================
# train_ensemble_v4.sh
# Deep Ensemble training for CT-to-MRI v4 — single-GPU SEQUENTIAL execution.
#
# Trains N=5 independent models, one after another.
# Each member uses a different random seed → different weight initialisation
# and different mini-batch order (controlled by set_random_seed in the script).
#
# Usage:
#   bash train_ensemble_v4.sh
#
# The experiment names will be:
#   ${BASE_EXP_NAME}_seed1337
#   ${BASE_EXP_NAME}_seed42
#   ${BASE_EXP_NAME}_seed123
#   ${BASE_EXP_NAME}_seed456
#   ${BASE_EXP_NAME}_seed789
#
# Checkpoints are saved to:
#   ${CKPT_ROOT}/${BASE_EXP_NAME}_seed<SEED>/<name>_epoch<NNN>.pt
# =============================================================================

set -e  # abort on first error

# ---- User configuration (edit these) ----------------------------------------
TRAIN_METADATA="/pscratch/sd/s/seojw/CT_to_MRI/metadata/train_metadata_total_n226_20260302.csv"
VAL_METADATA="/pscratch/sd/s/seojw/CT_to_MRI/metadata/val_metadata_total_n226_20260302.csv"
BASE_EXP_NAME="bbdm_ct2mri_SynthRAD+severance_mni_n226_v4_ensemble"

DROPOUT_RATE=0.0        # Set to 0.05 if you also want MC Dropout on same models
BATCH_SIZE=2
GRAD_ACCUM=8
MAX_EPOCHS=10000
WANDB_PROJECT="CT_to_MRI"
WANDB_ENTITY="connectome"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_bbdm_ct2mri_wandb_v4.py"

# ---- Five seeds (different init + data order per member) --------------------
SEEDS=(1337 42 123 456 789)

# ---- Environment (adjust for your cluster) ----------------------------------
module load python 2>/dev/null || true
conda activate flashattention 2>/dev/null || true

# =============================================================================
# Sequential training loop — ONE model on GPU at a time
# =============================================================================
N=${#SEEDS[@]}
echo "============================================================"
echo "Deep Ensemble Training: N=${N} members"
echo "Base experiment name : ${BASE_EXP_NAME}"
echo "Dropout rate         : ${DROPOUT_RATE}"
echo "============================================================"

for i in "${!SEEDS[@]}"; do
    SEED=${SEEDS[$i]}
    EXP_NAME="${BASE_EXP_NAME}_seed${SEED}"

    echo ""
    echo "------------------------------------------------------------"
    echo "Training member $((i + 1))/${N}  |  seed=${SEED}"
    echo "Experiment: ${EXP_NAME}"
    echo "------------------------------------------------------------"

    python "${TRAIN_SCRIPT}" \
        --train_metadata "${TRAIN_METADATA}" \
        --val_metadata   "${VAL_METADATA}"   \
        --experiment_name "${EXP_NAME}"      \
        --seed            "${SEED}"          \
        --dropout_rate    "${DROPOUT_RATE}"  \
        --wandb_project   "${WANDB_PROJECT}" \
        --wandb_entity    "${WANDB_ENTITY}"  \
        --batch_size      "${BATCH_SIZE}"    \
        --grad_accum_steps "${GRAD_ACCUM}"   \
        --max_epochs      "${MAX_EPOCHS}"    \
        --mni

    echo "Member $((i + 1))/${N} finished (seed=${SEED})."
done

echo ""
echo "============================================================"
echo "All ${N} ensemble members trained successfully."
echo "Checkpoints are in: /pscratch/sd/s/seojw/CT_to_MRI/checkpoints/"
echo "Next step: run generate_uq_v4.sh"
echo "============================================================"
