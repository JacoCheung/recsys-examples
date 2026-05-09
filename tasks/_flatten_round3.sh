#!/bin/bash
# Post-flatten round3 sweep results into 2-level structure.
#
# Before:
#   <SWEEP_ROOT>/<variant>/<timestamp>/<exp_name>/{*.gin,*.log,*.nsys-rep,*.out}
# After:
#   <SWEEP_ROOT>/<variant>/{*.gin,*.log,*.nsys-rep,*.out}
#
# Idempotent: if a variant dir is already flat (no subdirs), skipped.
set -euo pipefail

SWEEP_ROOT="${1:?usage: $0 <SWEEP_ROOT>}"

if [ ! -d "${SWEEP_ROOT}" ]; then
  echo "ERROR: ${SWEEP_ROOT} does not exist" >&2
  exit 1
fi

shopt -s nullglob
for variant_dir in "${SWEEP_ROOT}"/*/; do
  variant_dir="${variant_dir%/}"
  variant_name="$(basename "${variant_dir}")"

  # Skip the tracking file
  [ "${variant_name}" = "_track.txt" ] && continue
  [ ! -d "${variant_dir}" ] && continue

  # Find the deepest dir that holds files (timestamp/exp_name)
  exp_dirs=( "${variant_dir}"/*/*/ )
  if [ ${#exp_dirs[@]} -eq 0 ]; then
    echo "[skip] ${variant_name}: no nested dirs (already flat?)"
    continue
  fi
  if [ ${#exp_dirs[@]} -gt 1 ]; then
    echo "[warn] ${variant_name}: multiple nested experiment dirs:"
    printf '         %s\n' "${exp_dirs[@]}"
    echo "       (continuing — moving the first match)"
  fi

  src_dir="${exp_dirs[0]%/}"
  echo "[flatten] ${variant_name} <- ${src_dir}"
  # Move all files (no subdirs expected at exp level)
  mv "${src_dir}"/* "${variant_dir}/" 2>/dev/null || true
  # Clean up empty timestamp + exp dirs
  rmdir "${src_dir}" 2>/dev/null || true
  ts_dir="$(dirname "${src_dir}")"
  rmdir "${ts_dir}" 2>/dev/null || true

  echo "  -> $(ls "${variant_dir}" | wc -l) files in ${variant_dir}"
done

echo ""
echo "Flatten done. Final layout:"
ls -d "${SWEEP_ROOT}"/*/ 2>/dev/null | while read d; do
  printf "  %-50s (%d files)\n" "${d}" "$(ls "${d}" 2>/dev/null | wc -l)"
done
