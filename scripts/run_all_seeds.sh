#!/bin/bash
# Train ViT-UNet (4-skip + pos_weight 1.5) across 3 seeds on either dataset.
#
# Usage:
#   bash scripts/run_all_seeds.sh                      # refined dataset
#   bash scripts/run_all_seeds.sh original             # original Deep-SAR dataset
#
# Each seed runs to early-stopping (patience 15 on val_loss) or 100 epochs.
# Checkpoints + per-epoch history land in $SAVE_DIR with the matching tag.
set -e
cd "$(dirname "$0")/.."

DATASET="${1:-refined}"

if [ "$DATASET" = "original" ]; then
  DATA_PATH="./dataset_orig"
  SAVE_DIR="./models_results_orig"
  TAG="vit_unet_4skip_orig"
else
  DATA_PATH="./dataset"
  SAVE_DIR="./models_results"
  TAG="vit_unet_4skip"
fi

for seed in 42 123 7; do
  echo ""
  echo "=========================================="
  echo "  ${TAG} seed ${seed} on ${DATASET}"
  echo "=========================================="
  date
  python -u scripts/train.py \
    --model vit_unet_4skip --pos_weight 1.5 \
    --dataset "$DATASET" --data_path "$DATA_PATH" \
    --save_dir "$SAVE_DIR" --tag "$TAG" \
    --epochs 100 --early_stop_patience 15 \
    --seed ${seed} \
    > "${SAVE_DIR}/${TAG}_seed${seed}_train.log" 2>&1
  echo "Seed ${seed} DONE."
  date
done

echo ""
echo "=========================================="
echo "  All seeds done. Aggregating ..."
echo "=========================================="
python -u scripts/aggregate_4skip_seeds.py \
  --dataset "$DATASET" --data_path "$DATA_PATH" \
  --save_dir "$SAVE_DIR" --tag "$TAG"
