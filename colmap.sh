#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash colmap.sh <scene_dir> <datatype> [--force]

Supported datatype values correspond to scripts/<datatype>2colmap.py.
For N3DV, use: bash colmap.sh <scene_dir> llff

The script creates sparse_, image_colmap, and colmap inside <scene_dir>.
If any of those derived directories already exist, pass --force to replace them.
EOF
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage
  exit 1
fi

workdir=$1
datatype=$2
force=${3:-}
converter="scripts/${datatype}2colmap.py"

if [[ ! -d "$workdir" ]]; then
  echo "[error] Scene directory does not exist: $workdir" >&2
  exit 1
fi
if [[ ! -f "$converter" ]]; then
  echo "[error] Unsupported datatype '$datatype': missing $converter" >&2
  exit 1
fi
if [[ "$force" != "" && "$force" != "--force" ]]; then
  usage
  exit 1
fi

derived_dirs=("$workdir/sparse_" "$workdir/image_colmap" "$workdir/colmap")
existing=()
for path in "${derived_dirs[@]}"; do
  [[ -e "$path" ]] && existing+=("$path")
done

if (( ${#existing[@]} > 0 )); then
  if [[ "$force" != "--force" ]]; then
    echo "[error] Derived preprocessing outputs already exist:" >&2
    printf '  %s\n' "${existing[@]}" >&2
    echo "Re-run with --force only if replacing those derived outputs is intended." >&2
    exit 1
  fi
  rm -rf -- "${existing[@]}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python "$converter" "$workdir"

mkdir -p "$workdir/colmap"
cp -a "$workdir/image_colmap" "$workdir/colmap/images"
cp -a "$workdir/sparse_" "$workdir/colmap/sparse_custom"

colmap feature_extractor \
  --database_path "$workdir/colmap/database.db" \
  --image_path "$workdir/colmap/images" \
  --SiftExtraction.max_image_size 4096 \
  --SiftExtraction.max_num_features 16384 \
  --SiftExtraction.estimate_affine_shape 1 \
  --SiftExtraction.domain_size_pooling 1

python database.py \
  --database_path "$workdir/colmap/database.db" \
  --txt_path "$workdir/colmap/sparse_custom/cameras.txt"

colmap exhaustive_matcher --database_path "$workdir/colmap/database.db"

mkdir -p "$workdir/colmap/sparse/0"
colmap point_triangulator \
  --database_path "$workdir/colmap/database.db" \
  --image_path "$workdir/colmap/images" \
  --input_path "$workdir/colmap/sparse_custom" \
  --output_path "$workdir/colmap/sparse/0" \
  --clear_points 1

mkdir -p "$workdir/colmap/dense/workspace"
colmap image_undistorter \
  --image_path "$workdir/colmap/images" \
  --input_path "$workdir/colmap/sparse/0" \
  --output_path "$workdir/colmap/dense/workspace"
colmap patch_match_stereo --workspace_path "$workdir/colmap/dense/workspace"
colmap stereo_fusion \
  --workspace_path "$workdir/colmap/dense/workspace" \
  --output_path "$workdir/colmap/dense/workspace/fused.ply"

