#! /bin/bash

device=${1:-cuda:0}
for method in agreed pairwise
do
    for W in 16 20 24 28 32
    do
        python 1_run_AGREED.py --method=$method --W=$W --device=$device --save
    done
done
