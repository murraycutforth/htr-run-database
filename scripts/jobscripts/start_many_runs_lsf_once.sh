#!/bin/bash

start=$1
end=$2

for i in $(seq $start $end)
do
    padded_i=$(printf "%04d" $i)
    echo "Starting run $padded_i"
    cd $padded_i || exit
    ./run-htr-once_lsf.sh &
    sleep 0.1
    cd ..
done
