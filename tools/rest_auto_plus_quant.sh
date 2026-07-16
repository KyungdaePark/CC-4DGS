#!/usr/bin/env bash
set -euo pipefail

# Orchestrate REST autoencoder training + rest-free quantisation + compressed render/metrics.
# Usage:
#   ./tools/rest_auto_plus_quant.sh \
#       --model output/.../scene/runX \
#       --source-iter 20000 \
#       --iteration 20000 \
#       --dataset /path/to/N3DV/coffee_martini \
#       --config arguments/dynerf/coffee_martini.py \
#       [additional rest_autoencoder.py flags...]

MODEL_PATH=""
SOURCE_ITER=""
TARGET_ITER=""
DATASET_PATH=""
CONFIG_PATH=""
RESTFREE_GROUPS="scale,opacity,rotation,dynamic"
RUN_RENDER=1
RUN_METRICS=1
CLEANUP_AUTOENCODER=0
RENDER_CONDA_ENV="${RENDER_CONDA_ENV:-}"
CONDITION_FIELDS=(x y z opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3 dynamic)
AUTO_ARGS=(
  --latent-dim 24
  --hidden-dim 512
  --num-layers 4
  --dropout 0.05
  --batch-size 16384
  --steps 50000
  --lr 5e-4
  --device cuda
  --log-every 200
  --dtype float16
  --chunk -1
  --quant-bits 14
  --latents-compression npz_zstd
  --bitpack-latents
  --checkpoint-dtype float16
)
AUTO_ARGS+=(--condition-fields "${CONDITION_FIELDS[@]}")
AUTO_ARGS+=(--residual-codebook-size 256 --residual-iters 20 --residual-condition-weight 0.2)
usage() {
  cat <<'EOF'
Usage: tools/rest_auto_plus_quant.sh --model <path> --source-iter <iter> --iteration <iter> \
       --dataset <path> --config <path> [--quant-groups groups] [rest_autoencoder.py flags...]
Optional flags:
  --quant-groups <csv>   Comma-separated groups for rest-free quantisation (default: scale,opacity,rotation,dynamic)
  --no-render            Skip render.py --compressed
  --no-metrics           Skip metrics.py --compressed
  --cleanup-autoencoder  Remove model_path/autoencoder/iteration_<iter> after a successful run
  --keep-autoencoder     Preserve autoencoder artifacts (default)
  --                     Treat the remaining arguments literally (forward to rest_autoencoder.py)
All unknown flags before '--' are forwarded to rest_autoencoder.py auto command.
EOF
}

run_eval_python() {
  if [[ -n "$RENDER_CONDA_ENV" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
      echo "[error] RENDER_CONDA_ENV is set, but conda is not available on PATH." >&2
      exit 1
    fi
    conda run --no-capture-output -n "$RENDER_CONDA_ENV" python "$@"
  else
    python "$@"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model|--model-path)
      MODEL_PATH="$2"; shift 2 ;;
    --source-iter)
      SOURCE_ITER="$2"; shift 2 ;;
    --iteration)
      TARGET_ITER="$2"; shift 2 ;;
    --dataset|--dataset-path)
      DATASET_PATH="$2"; shift 2 ;;
    --config|--config-path)
      CONFIG_PATH="$2"; shift 2 ;;
    --quant-groups)
      RESTFREE_GROUPS="$2"; shift 2 ;;
    --no-render)
      RUN_RENDER=0; shift ;;
    --no-metrics)
      RUN_METRICS=0; shift ;;
    --cleanup-autoencoder)
      CLEANUP_AUTOENCODER=1; shift ;;
    --keep-autoencoder)
      CLEANUP_AUTOENCODER=0; shift ;;
    --help|-h)
      usage; exit 0 ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        AUTO_ARGS+=("$1")
        shift
      done
      break ;;
    *)
      AUTO_ARGS+=("$1")
      shift ;;
  esac
done

if [[ -z "$MODEL_PATH" || -z "$SOURCE_ITER" || -z "$TARGET_ITER" || -z "$DATASET_PATH" || -z "$CONFIG_PATH" ]]; then
  usage
  exit 1
fi

python tools/rest_autoencoder.py auto \
  --model-path "$MODEL_PATH" \
  --source-iteration "$SOURCE_ITER" \
  --iteration "$TARGET_ITER" \
  --deploy-iteration "$TARGET_ITER" \
  --write-compressed \
  --restfree-groups "$RESTFREE_GROUPS" \
  "${AUTO_ARGS[@]}"

python - <<PY
from pathlib import Path
import sys

model = Path("$MODEL_PATH")
source_iter = int("$SOURCE_ITER")
target_iter = int("$TARGET_ITER")
groups = "$RESTFREE_GROUPS"

src_ply = model / "point_cloud" / f"iteration_{source_iter}" / "point_cloud.ply"
orig_bytes = src_ply.stat().st_size if src_ply.exists() else 0

comp_dir = model / "point_cloud" / f"iteration_{target_iter}" / "compressed"
if not comp_dir.exists():
    print(f"[summary] Warning: compressed dir missing at {comp_dir}, skipping size report.")
    sys.exit(0)

def sum_files(root: Path) -> int:
    if not root.exists():
        return 0
    if root.is_file():
        return root.stat().st_size
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total

rest_files = ["latents.npz", "rest_autoencoder.pt", "residual_codebook.npy", "residual_assignments.npy"]
rest_bytes = sum((comp_dir / name).stat().st_size for name in rest_files if (comp_dir / name).exists())

restfree_dir = comp_dir / "restfree"
restfree_bytes = sum_files(restfree_dir)
manifest = comp_dir / "restfree_manifest.json"
if manifest.exists():
    restfree_bytes += manifest.stat().st_size

canonical_bytes = rest_bytes + restfree_bytes
iteration_dir = model / "point_cloud" / f"iteration_{target_iter}"
model_state_names = ("deformation.pth", "FDHash.pth")
deform_bytes = sum(
    (iteration_dir / name).stat().st_size
    for name in model_state_names
    if (iteration_dir / name).exists()
)
deployable_bytes = canonical_bytes + deform_bytes

def fmt_size(bytes_val: int) -> str:
    return (
        f"{bytes_val / 1_000_000:.2f} MB "
        f"({bytes_val / (1024 * 1024):.2f} MiB; {bytes_val:,} bytes)"
    )

if orig_bytes == 0:
    print(f"[summary] Original PLY missing at {src_ply}; showing compressed sizes only.")
    print(f"[summary] REST payload: {fmt_size(rest_bytes)}")
    print(f"[summary] Rest-free payload: {fmt_size(restfree_bytes)} (groups: {groups})")
else:
    def pct(val: int) -> float:
        return 100.0 * val / orig_bytes if orig_bytes else 0.0
    print(f"[summary] Original point_cloud.ply: {fmt_size(orig_bytes)}")
    print(f"[summary] REST payload (latents/checkpoint/residual): {fmt_size(rest_bytes)} -> {pct(rest_bytes):.1f}% of original")
    print(f"[summary] Rest-free payload (groups {groups}): {fmt_size(restfree_bytes)} -> {pct(restfree_bytes):.1f}% of original")

print(f"[summary] CCA canonical-attribute payload: {fmt_size(canonical_bytes)}")
print(f"[summary] Deform-module storage: {fmt_size(deform_bytes)}")
print(f"[summary] Total deployable model size: {fmt_size(deployable_bytes)}")
PY

if [[ $RUN_RENDER -eq 1 ]]; then
  run_eval_python render.py \
    -s "$DATASET_PATH" \
    --configs "$CONFIG_PATH" \
    --model_path "$MODEL_PATH" \
    --iteration "$TARGET_ITER" \
    --skip_video \
    --skip_train \
    --compressed
fi

if [[ $RUN_METRICS -eq 1 ]]; then
  run_eval_python metrics.py \
    -m "$MODEL_PATH" \
    --iteration "$TARGET_ITER" \
    --compressed
fi

if [[ $CLEANUP_AUTOENCODER -eq 1 ]]; then
  auto_dir="$MODEL_PATH/autoencoder/iteration_${TARGET_ITER}"
  if [[ -d "$auto_dir" ]]; then
    echo "[cleanup] Removing autoencoder artifacts at $auto_dir"
    rm -rf "$auto_dir"
  else
    echo "[cleanup] autoencoder directory not found at $auto_dir (nothing to remove)"
  fi
  # Drop the parent autoencoder dir if empty to avoid clutter.
  parent_dir="$MODEL_PATH/autoencoder"
  if [[ -d "$parent_dir" && -z "$(ls -A "$parent_dir")" ]]; then
    rmdir "$parent_dir"
  fi
fi
