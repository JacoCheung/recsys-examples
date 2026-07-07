#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUTPUT_DIR=${PROFILE_OUTPUT_DIR:?Set PROFILE_OUTPUT_DIR to the profile result path}
REPORTS_DIR="${OUTPUT_DIR}/reports"

for report in "${REPORTS_DIR}"/*.nsys-rep; do
    nsys export \
        --type=sqlite \
        --force-overwrite=true \
        --output="${report%.nsys-rep}.sqlite" \
        "${report}"
done

python3 "${SCRIPT_DIR}/validate_hstu_attn_nsys_reports.py" "${REPORTS_DIR}" \
    > "${OUTPUT_DIR}/report_validation.tsv"
