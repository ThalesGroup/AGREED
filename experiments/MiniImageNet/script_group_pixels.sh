#! /bin/bash

device="cuda:0"
num_threads=2
for model in ResNet18 vgg16
do
    for method in agreed square quick
    do
        python 1_run_AGREED.py --method=$method --model=$model --device=$device --num_threads=$num_threads --init_patch_size=14
    done
done

# For ConvNext, the first layer considers 4x4 disjoint patches so we use 16x16
for method in agreed square quick
do
    python 1_run_AGREED.py --method=$method --model=$model --device=$device --num_threads=$num_threads --init_patch_size=16
done
