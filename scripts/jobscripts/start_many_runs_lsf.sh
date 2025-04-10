#!/bin/bash

start=$1
end=$2
MAX_ITER=$3

for i in $(seq $start $end)
do
    padded_i=$(printf "%04d" $i)
    echo "Starting run $padded_i"
    cd $padded_i || exit
    ./run-htr-with-restarts_lsf.sh "${MAX_ITER}" &
    sleep 0.1
    cd ..
done
