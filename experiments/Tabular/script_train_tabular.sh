# !/bin/bash

if [ -z "$1" ]
then 
    echo "Please pass a positional arg"
    echo "script_train_tabular.sh <dataset>"
    exit 1
fi
dataset=$1

python3 0_train_model.py --dataset=$dataset --model=ebm --n_jobs=8
python3 0_train_model.py --dataset=$dataset --model=rf --n_jobs=8 --n_repetitions=40
python3 0_train_model.py --dataset=$dataset --model=gbt --n_jobs=8 --n_repetitions=100
