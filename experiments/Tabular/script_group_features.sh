# !/bin/bash

[ "$#" -lt 1 ] && echo "script_group_features.sh <seed>" && exit 1

seed=$1

# Credit
for model in ebm gbt
do
    for method in agreed #recurse igreedy pairwise
    do
        python3 1_run_AGREED.py --dataset=default_credit --model=$model --n_samples=100 --n_groups_min=14 --method=$method --seed=$seed --save
    done
    #  python3 1_run_baseline.py --dataset=default_credit --model=$model --n_samples=100 --seed=$seed --n_groups_min=14 --save
done

# Marketing
for model in ebm gbt
do
    for method in agreed #recurse igreedy pairwise
    do
        python3 1_run_AGREED.py --dataset=marketing --model=$model --n_samples=100 --n_groups_min=10 --method=$method --seed=$seed --save
    done
    #  python3 1_run_baseline.py --dataset=marketing --model=$model --n_samples=100 --seed=$seed --n_groups_min=10 --save
done

# SPAM
for model in ebm gbt
do
    for method in agreed #recurse
    do
        python3 1_run_AGREED.py --dataset=spam --model=$model --n_samples=50 --n_groups_min=40 --method=$method --seed=$seed --save
    done
    #  python3 1_run_baseline.py --dataset=spam --model=$model --n_samples=50 --seed=$seed --n_groups_min=40 --save
done


# # NOMAO (only AGREED and GBT scale)
python3 1_run_AGREED.py --dataset=nomao --model=gbt --n_samples=50 --seed=$seed --n_groups_min=100 --method=agreed --save
#  python3 1_run_baseline.py --dataset=nomao --model=gbt --n_samples=50 --seed=$seed --n_groups_min=100 --save
