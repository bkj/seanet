#!/bin/bash

# run-test.sh

RUN_NAME="run-n12-m2-s20"
find ../results/models/$RUN_NAME -type f |\
    xargs -I {} python ./test.py --inpath {} --outpath {}.predictions

cd ../results/models/$RUN_NAME 
mkdir ../../predictions/$RUN_NAME
mv *npy ../../predictions/$RUN_NAME