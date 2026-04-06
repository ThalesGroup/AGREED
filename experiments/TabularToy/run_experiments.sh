# !/bin/bash

if [ -z "$1" ]
then
    echo "Provide a positional argument"
    echo "run_experiments.sh <random_seed>"
    exit 1
fi

seed=$1
for d in 10 25 50 75 100
do
    for rho in 0 0.5 0.95
    do
        for N in 10 20 30 40 50
        do
            for method in agreed pairwise recurse igreedy
            do
                python3 1_run_AGREED.py --method=$method --d=$d --n_samples=$N --rho=$rho --seed=$seed --save
            done
        done
    done
done
