#!/bin/bash
# Train two-stage NCCT segmentation pipeline
#
# Stage 1: Train slice-level stroke detector to identify stroke-positive slices.
# Stage 2: Train segmentation U-Net, feeding only stroke-positive slices.
#
# Usage:
#   bash tools/train_two_stage.sh <data_root> [config] [gpus]
#
# Examples:
#   bash tools/train_two_stage.sh data/ncct
#   bash tools/train_two_stage.sh data/ncct configs/ncct/unet_fcn_stroke_ncct.py 2
#   bash tools/train_two_stage.sh /kaggle/working/data/ncct configs/ncct/deeplabv3plus_r50_stroke_ncct.py 2

set -euo pipefail

DATA_ROOT="${1:?Usage: $0 <data_root> [config] [gpus]}"
CONFIG="${2:-configs/ncct/unet_fcn_stroke_ncct.py}"
GPUS="${3:-1}"
WORK_DIR="$(dirname "$CONFIG" | sed 's|configs/|work_dirs/|')/$(basename "$CONFIG" .py)"
DETECTOR_DIR="${DATA_ROOT}/detector"

echo "=========================================="
echo "Two-Stage NCCT Segmentation Pipeline"
echo "=========================================="
echo "Data root:   $DATA_ROOT"
echo "Config:      $CONFIG"
echo "GPUs:        $GPUS"
echo "Work dir:    $WORK_DIR"
echo "Detector:    $DETECTOR_DIR"
echo "=========================================="
echo ""

# ── Stage 1: Train Detector ──
echo "[Stage 1] Training slice-level stroke detector..."
echo "=========================================="

python tools/stage1_detector.py "$DATA_ROOT" \
    --work-dir "$DETECTOR_DIR" \
    --epochs 50 \
    --batch-size 64 \
    --backbone resnet18

# Check detector results
POS_LIST="${DETECTOR_DIR}/stroke_positive_train.txt"
if [ ! -f "$POS_LIST" ]; then
    echo "ERROR: Detector did not produce stroke_positive_train.txt"
    exit 1
fi

N_POS=$(wc -l < "$POS_LIST")
TOTAL=$(ls "$DATA_ROOT/train/images" 2>/dev/null | grep -c '\.png$' || echo 0)
echo ""
echo "[Stage 1] Detector complete: $N_POS / $TOTAL training slices are stroke-positive"
echo ""

# ── Stage 2: Train Segmentation on stroke-positive slices ──
echo "[Stage 2] Training segmentation model on stroke-positive data..."
echo "=========================================="

# Build cfg-options to filter training data and use the new loss
mkdir -p "$WORK_DIR"

python tools/train.py "$CONFIG" \
    --work-dir "$WORK_DIR" \
    --amp \
    ${GPUS} \
    --cfg-options \
        data_root="${DATA_ROOT}/" \
        train_dataloader.dataset.stroke_positive_only=True \
        train_dataloader.dataset.stroke_positive_file="${POS_LIST}"

echo ""
echo "=========================================="
echo "Two-stage training complete!"
echo "=========================================="
echo "Detector:  $DETECTOR_DIR/best_detector.pth"
echo "Segmentation: $WORK_DIR"
echo ""
echo "Next steps:"
echo "  python tools/test.py $CONFIG \$WORK_DIR/iter_*.pth --show-dir \$WORK_DIR/test_preds"
echo "=========================================="
