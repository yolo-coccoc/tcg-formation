#!/bin/bash

# Script to run MERRA2 training job with default parameters
# Usage: ./run_merra2_train.sh [OPTIONS]
# Example: ./run_merra2_train.sh --project my_project --max_epochs 50

set -e

# ==================== Configuration ====================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/merra2_train_job_base.py"

# Conda environment path
CONDA_ENV_PATH="/N/slate/tnn3/hoa/diffusion"

# Default parameters (override with command-line arguments)
PROJECT="test"
SEED=42
WEIGHT=6.0
BATCH_SIZE=64
NUM_WORKERS=2
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01
MAX_EPOCHS=2
MODE=-1
INP_DIR="./"
OUT_DIR="./"
EXPORT_RESULT=""

# ==================== Functions ====================
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    --project             WandB project name (default: test)
    --seed                Random seed (default: 42)
    --weight              Positive class weight (default: 6.0)
    --batch_size          Batch size (default: 64)
    --num_workers         Number of workers (default: 2)
    --learning_rate       Learning rate (default: 1e-4)
    --weight_decay        Weight decay (default: 0.01)
    --max_epochs          Maximum epochs (default: 2)
    --mode                Mode: -1 for new run, >=0 for resume (default: -1)
    --inp_dir             Input directory (default: ./)
    --out_dir             Output directory (default: ./)
    --export_result       Export result file name (optional)
    --help                Show this help message

Example:
    $0 --project tc_prediction --max_epochs 100 --batch_size 32
EOF
}

# ==================== Parse Arguments ====================
while [[ $# -gt 0 ]]; do
    case $1 in
        --project)
            PROJECT="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --weight)
            WEIGHT="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --weight_decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        --max_epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --inp_dir)
            INP_DIR="$2"
            shift 2
            ;;
        --out_dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --export_result)
            EXPORT_RESULT="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# ==================== Setup Environment ====================
echo "================================================"
echo "Setting up environment..."
echo "================================================"

# Source conda
source /N/soft/sles15sp6/conda/25.3.0/etc/profile.d/conda.sh

# Activate environment
if [ -d "$CONDA_ENV_PATH" ]; then
    conda activate "$CONDA_ENV_PATH"
    echo "✓ Conda environment activated: $CONDA_ENV_PATH"
else
    echo "✗ Error: Conda environment not found at $CONDA_ENV_PATH"
    exit 1
fi

# Verify Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "✗ Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi
echo "✓ Python script found: $PYTHON_SCRIPT"

# ==================== Print Configuration ====================
echo ""
echo "================================================"
echo "Training Configuration"
echo "================================================"
echo "Project:           $PROJECT"
echo "Seed:              $SEED"
echo "Weight:            $WEIGHT"
echo "Batch Size:        $BATCH_SIZE"
echo "Num Workers:       $NUM_WORKERS"
echo "Learning Rate:     $LEARNING_RATE"
echo "Weight Decay:      $WEIGHT_DECAY"
echo "Max Epochs:        $MAX_EPOCHS"
echo "Mode:              $MODE"
echo "Input Dir:         $INP_DIR"
echo "Output Dir:        $OUT_DIR"
if [ -n "$EXPORT_RESULT" ]; then
    echo "Export Result:     $EXPORT_RESULT"
fi
echo "================================================"
echo ""

# ==================== Run Training ====================
echo "Starting training..."
python "$PYTHON_SCRIPT" \
    --project "$PROJECT" \
    --seed "$SEED" \
    --weight "$WEIGHT" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay "$WEIGHT_DECAY" \
    --max_epochs "$MAX_EPOCHS" \
    --mode "$MODE" \
    --inp_dir "$INP_DIR" \
    --out_dir "$OUT_DIR" \
    $([ -n "$EXPORT_RESULT" ] && echo "--export_result $EXPORT_RESULT")

EXIT_CODE=$?

# ==================== Cleanup ====================
echo ""
echo "================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully!"
else
    echo "✗ Training failed with exit code: $EXIT_CODE"
fi
echo "================================================"

exit $EXIT_CODE
