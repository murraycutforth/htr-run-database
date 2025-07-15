#!/bin/bash
# This script is used to run the HTR code with no restarts
# It assumes that run-htr.sh, prepare_restart_run.py, GG-combustor.json all exists in the same directory
# ============================================================

RUN_COMMAND="./run-htr.sh"


# Function to check job status and find the latest checkpoint
check_job_status() {
    local job_id=$1  # Get the job ID from the first argument
    sleep 60

    while true; do
        # Check if the job is still running or in the queue
        job_info=$(bjobs -noheader $job_id)

        if [[ "$job_info" == *"is not found"* ]]; then
            break
        else
            # Job is still queueing or running
            job_status=$(echo "$job_info" | awk '{print $3}')

            if [[ "$job_status" == "DONE" ]]; then
                break
            elif [[ "$job_status" == "EXIT" ]]; then
                break
            fi
            sleep 60
        fi
    done
}


# Function to extract the latest checkpoint file
find_latest_checkpoint() {
    latest_checkpoint=$(find . -type d -wholename "./sample0/fluid_iter*" | sort | tail -n 1)

    if [[ -n "$latest_checkpoint" ]]; then
        iterations=$(echo "$latest_checkpoint" | grep -o 'fluid_iter[0-9]*' | grep -o '[0-9]*')

        # Remove leading zeros from the iterations
        iterations=$(echo "$iterations" | sed 's/^0*//')

        # Handle case where all zeros are stripped
        if [[ -z "$iterations" ]]; then
            iterations=0
        fi

        echo "$iterations"
    else
        return 1
    fi
}


# =================
# Main script
# =================

job_sub_msg=$($RUN_COMMAND)
job_id=$(echo "$job_sub_msg" | grep -oP '(?<=Job <)\d+(?=>)')
check_job_status $job_id  # We remain in this function until job is no longer in RUN or PEND state
latest_checkpoint=$(find_latest_checkpoint)
python3 prepare_restart_run.py $latest_checkpoint





