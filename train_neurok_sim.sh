#!/usr/bin/env bash
set -euo pipefail

data_root="/svl/u/yuegao/NeuROK/PhysDreamer/phys_dreamer/data_NeuROK_sim"


# all_names = bow box cloth flower newton
all_names=(cloth)

ITER=30000

# Optional: set to 1 to enable white background blending for RGBA PNGs
WHITE_BKGD=1

for object_name in "${all_names[@]}"; do
  output_folder=${object_name}_gs
  DATASET_DIR="$data_root/$object_name"
  MODEL_DIR="$data_root/$output_folder"

  echo "[train] object=$object_name dataset=$DATASET_DIR output=$MODEL_DIR"
  mkdir -p "$MODEL_DIR"

  OPTS=(
    -s "$DATASET_DIR"
    -m "$MODEL_DIR"
    --iterations "$ITER"
  )

  if [[ "$WHITE_BKGD" == "1" ]]; then
    OPTS+=( --white_background )
  fi

  python train.py "${OPTS[@]}"
  echo "[done] $object_name → $MODEL_DIR"
done
