#!/usr/bin/env bash
set -euo pipefail

data_root="/svl/u/yuegao/NeuROK/PhysDreamer/phys_dreamer/data_NeuROK_sim"
# all_names=(bow box cloth flower newton shirt lamp)
all_names=(laptop)


# "${1:?Usage: $0 DATASET_DIR}"

for object_name in "${all_names[@]}"; do
  DATASET_DIR="$data_root/$object_name"
  echo "[prep] object=$object_name dataset=$DATASET_DIR"
  if [[ -f "$DATASET_DIR/transforms.json" ]]; then
    python "$(dirname "$0")/prepare_transforms.py" \
      --dataset_dir "$DATASET_DIR" \
      --source transforms.json \
      --train_out transforms_train.json \
      --test_out transforms_test.json
  else
    echo "[warn] $DATASET_DIR/transforms.json not found; nothing to normalize."
  fi
  echo "[done] $object_name normalization step complete."
done
