#!/bin/bash
# This script is used to download the 2M grid sweep data from Lassen

set -e

USER=cutforth1
HOST=lassen.llnl.gov
BATCH=$1
RUN_BATCH_DIR="/p/gpfs1/cutforth1/PSAAP/runs_batch_${BATCH}/[0-9][0-9][0-9]*"

# Check that BATCH is 4-digit number
if [[ $BATCH =~ ^[0-9]{4}$ ]]; then
    echo "Valid 4-digit batch number"
else
    echo "Invalid batch number - must be exactly 4 digits"
    exit 1
fi

rsync -av --include='*/' --include='*/solution/*.out' --include='*/solution/*.csv' --include='*/solution/console-*.txt' --exclude '*' --rsync-path="env -i rsync" ${USER}@${HOST}:${RUN_BATCH_DIR} /Users/murray/Downloads/runs_batch_${BATCH}
python -m src.output_utils.main_assemble_pressure_trace_matrix --run-dir /Users/murray/Downloads/runs_batch_${BATCH}
python -m src.output_utils.main_assemble_ignition_outcome --run-dir /Users/murray/Downloads/runs_batch_${BATCH}
