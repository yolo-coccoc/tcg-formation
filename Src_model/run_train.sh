#!/bin/bash

# Activate conda environment
# source /N/soft/sles15sp6/conda/25.3.0/etc/profile.d/conda.sh
# conda activate /N/slate/tnn3/hoa/diffusion

# Default arguments
PROJECT="test"
SEED=42
WEIGHT=6.0
BATCH_SIZE=64
NUM_WORKERS=2
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01
MAX_EPOCHS=2
MODE=0
CHECKPOINT="last"
INP_DIR="/N/slate/tnn3/DucHGA/TC-formation/Data/Merra/Dataset/Rus10"
OUT_DIR="/N/slate/tnn3/DucHGA/TC-formation/OutTemp/test"

# Run training script
python /N/slate/tnn3/DucHGA/TC-formation/Src_model/merra2_train_job_base.py \
    --project "$PROJECT" \
    --seed "$SEED" \
    --weight "$WEIGHT" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay "$WEIGHT_DECAY" \
    --max_epochs "$MAX_EPOCHS" \
    --mode "$MODE" \
    --checkpoint "$CHECKPOINT" \
    --inp_dir "$INP_DIR" \
    --out_dir "$OUT_DIR"
