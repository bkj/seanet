#!/bin/bash

# run.sh

python main.py --run-name val-0.8-v --train-size 0.8 # Using validation error for model search
python main.py --run-name val-0.8-v2 --train-size 0.8 # + breaking when a model diverges


python main.py --run-name val-0.8-v2 --train-size 0.8 # + breaking when a model diverges

# Different parameters
python main.py --run-name run-n12-m2-s20 --train-size 0.8 \
    --num-neighbors 12 \
    --num-morphs 2 \
    --num-steps 20
