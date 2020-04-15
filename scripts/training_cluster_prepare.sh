#!/bin/bash

SCRIPTPATH=$(dirname $(realpath $0))

# Source parameters
source $SCRIPTPATH/training_cluster_params.sh

# Create the training dir
rm -rf $TRAIN_DIR
mkdir -p $TRAIN_DIR

# Copy the model configuration files
MODEL_CONFIG_DIR=$CODE_DIR/configs/$MODEL
cp $MODEL_CONFIG_DIR/vocab.txt $TRAIN_DIR/vocab.txt
cp $MODEL_CONFIG_DIR/bert_config.json $TRAIN_DIR/bert_config.json

# Iterate through configs (Sequence Length, Batch)
STAGE=0
for CONFIG in $CONFIGS; do

  IFS=","
  set -- $CONFIG

  #SEQ=$1
  #BATCH=$2
  #CUR_TRAIN_DIR=$TRAIN_DIR/seq${SEQ}_ba${BATCH}_step$STEPS
  CUR_TRAIN_DIR=$TRAIN_DIR/Pretrain_Stage_$STAGE
  echo prep $CUR_TRAIN_DIR
  let STAGE+=1

  rm -rf $CUR_TRAIN_DIR
  mkdir -p $CUR_TRAIN_DIR

done
