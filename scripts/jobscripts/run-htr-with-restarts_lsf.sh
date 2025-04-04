#!/bin/bash
# This script is used to run the HTR code with restarts
# It assumes that run-htr.sh, prepare_restart_run.py, GG-combustor.json all exists in the same directory
# High level operation:
# 1. Submit job via run-htr.sh and record job id
# 2. For each restart:
 # a. Continuously monitor job status via bjobs until job has completed
 # b. If job has completed all iters, end script
 # d. If job has crashed (there is a random bug with the current HTR/Legion version), extract last checkpoint file and call prepare_restart_run.py in order to move output and update GG-combustor.json
 # e. Submit job again via run-htr.sh and record job id
# 3. Once all restarts have been completed, end script
# Logging of all steps from this script goes to run-htr-with-restarts.log
# ============================================================

NUM_RESTARTS=1
RUN_COMMAND="./run-htr.sh"
MAX_ITER=36000  # Don't restart if we have done \ge MAX_ITER iterations
LOGFILE="run-htr-with-restarts.log"
touch $LOGFILE


# Function to check job status and find the latest checkpoint
check_job_status() {
    local job_id=$1  # Get the job ID from the first argument

    while true; do
        # Check if the job is still running or in the queue
        job_info=$(bjobs -noheader $job_id)

        if [[ "$job_info" == *"is not found"* ]]; then
            echo "Job $job_id is not found. It may have completed or crashed. Checking for latest checkpoint..." >> $LOGFILE
            break
        else
            # Job is still queueing or running
            job_status=$(echo "$job_info" | awk '{print $3}')

            if [[ "$job_status" == "DONE" ]]; then
                echo "Job $job_id has completed normally. Checking for latest checkpoint..." >> $LOGFILE
                break
            elif [[ "$job_status" == "EXIT" ]]; then
                echo "Job $job_id has abnormal completion. Checking for latest checkpoint..." >> $LOGFILE
                break
            fi

            echo "Job $job_id has status ${job_status}... Checking again in 60 seconds." >> $LOGFILE
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

        echo "$iterations"
    else
        echo "No checkpoints found." >> $LOGFILE
        return 1
    fi
}


# =================
# Main script
# =================

job_sub_msg=$($RUN_COMMAND)
job_id=$(echo "$job_sub_msg" | grep -oP '(?<=Job <)\d+(?=>)')

echo "Job submitted with first ID $job_id" >> $LOGFILE

for i in $(seq 1 $NUM_RESTARTS); do
    check_job_status $job_id

    # TODO: check for crash and end early here if so

    latest_checkpoint=$(find_latest_checkpoint)

    if [[ $? -eq 1 ]]; then
        echo "No checkpoint found. Exiting script." >> $LOGFILE
        exit 1
    fi

    echo "Latest checkpoint found at iteration $latest_checkpoint" >> $LOGFILE

    if [[ $latest_checkpoint -ge $MAX_ITER ]]; then
        echo "Maximum number of iterations $MAX_ITER reached. Exiting restart loop." >> $LOGFILE
        break
    fi

    python3 prepare_restart_run.py $latest_checkpoint

    if [[ $? -eq 1 ]]; then
        echo "Error in prepare_restart_run.py. Exiting script." >> $LOGFILE
        exit 1
    fi

    job_sub_msg=$($RUN_COMMAND)
    job_id=$(echo "$job_sub_msg" | grep -oP '(?<=Job <)\d+(?=>)')

    echo "Job resubmitted with ID $job_id" >> $LOGFILE

done

check_job_status $job_id

echo "Maximum number of restarts reached. Finishing up" >> $LOGFILE

# Re-organise files from the last run so everything is in solution

latest_checkpoint=$(find_latest_checkpoint)

echo "Final checkpoint found at iteration $latest_checkpoint" >> $LOGFILE

if [[ $? -eq 1 ]]; then
    echo "No checkpoint found. Exiting script." >> $LOGFILE
    exit 1
fi

echo "Organising final set of checkpoints into solution dir" >> $LOGFILE

python3 prepare_restart_run.py $latest_checkpoint

echo "Script completed normally" >> $LOGFILE





