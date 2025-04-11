#!/bin/bash
# Copy a batch of console output files to current directory on local machine for analysis
# ===========

USER=cutforth1
HOST=lassen.llnl.gov
RUN_BATCH_DIR=/p/gpfs1/cutforth1/PSAAP/runs_batch_0001_0-199

rsync -av --include='*/' --include='*/solution/*.out' -include='*/solution/*.csv' --include='*/solution/console-*.txt' --exclude '*' --rsync-path="env -i rsync" ${USER}@${HOST}:${RUN_BATCH_DIR} .