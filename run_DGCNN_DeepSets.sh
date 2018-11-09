#!/bin/bash

# input arguments
DATA="${1-NCI1}"  # MUTAG, ENZYMES, NCI1, NCI109, DD, PTC, PROTEINS, COLLAB, IMDBBINARY, IMDBMULTI
fold=${2-1}  # which fold as testing data
test_number=${3-0}  # if specified, use the last test_number graphs as test data

# general settings
gm=DGCNNDS  # model
gpu_or_cpu=gpu
GPU=0  # select the GPU number
CONV_SIZE="32-32-32"
sortpooling_k=0.6  # If k <= 1, then k is set to an integer so that k% of graphs have nodes less than this integer
FP_LEN=128  # NICK: this is an hyperparameter we can tune final dense layer's input dimension, decided by data
n_hidden=128  # final dense layer's hidden size
bsize=50  # batch size
dropout=True

# dataset-specific settings
case ${DATA} in
MUTAG)
  num_epochs=300
  num_patience=50
  ;;
ENZYMES)
  num_epochs=500
  num_patience=50
  ;;
NCI1)
  num_epochs=200
  num_patience=50
  ;;
NCI109)
  num_epochs=200
  num_patience=50
  ;;
DD)
  num_epochs=200
  num_patience=50
  ;;
PTC)
  num_epochs=200
  num_patience=50
  ;;
PROTEINS)
  num_epochs=100
  num_patience=50
  ;;
COLLAB)
  num_epochs=300
  num_patience=50
  sortpooling_k=0.9
  ;;
IMDBBINARY)
  num_epochs=300
  sortpooling_k=0.9
  num_patience=50
  ;;
IMDBMULTI)
  num_epochs=500
  sortpooling_k=0.9
  num_patience=50
  ;;
*)
  num_epochs=500
  num_patience=50
  ;;
esac

CUDA_VISIBLE_DEVICES=${GPU} python3 main.py \
    -seed 1 \
    -data $DATA \
    -num_epochs $num_epochs \
    -num_patience $num_patience \
    -hidden $n_hidden \
    -latent_dim $CONV_SIZE \
    -sortpooling_k $sortpooling_k \
    -out_dim $FP_LEN \
    -batch_size $bsize \
    -gm $gm \
    -mode $gpu_or_cpu \
    -dropout $dropout



