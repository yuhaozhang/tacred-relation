#!/bin/bash

# This is an example script of training and running model ensembles.

# train 5 models with different seeds
python train.py --seed 0 --id 00
python train.py --seed 1 --id 01
python train.py --seed 2 --id 02
python train.py --seed 3 --id 03
python train.py --seed 4 --id 04

# evaluate on test sets and save prediction files
python eval.py saved_models/00 --out saved_models/out/test_0.pkl
python eval.py saved_models/01 --out saved_models/out/test_1.pkl
python eval.py saved_models/02 --out saved_models/out/test_2.pkl
python eval.py saved_models/03 --out saved_models/out/test_3.pkl
python eval.py saved_models/04 --out saved_models/out/test_4.pkl

# run ensemble
ARGS=""
for id in 1 2 3 4 5; do
    OUT="saved_models/out/test_${id}.pkl"
    ARGS="$ARGS $OUT"
done
python ensemble.py --dataset test $ARGS

