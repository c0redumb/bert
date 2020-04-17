#!/bin/bash

###################################
###      General Parameters     ###
###################################

# Model
MODEL=bert_large

# Directories
CODE_DIR=$SCRIPTPATH/..
TRAIN_DIR=/data/run_tmp/train_$MODEL

###################################
### Parameters for Pre-training ###
###################################

# Note: Unset the following to run with real data
TEST=1

# Leanring rate
LRN_RT=5e-5

# LM Probability
LM_PROB=0.15

# Function to calculate max_perdictions_per_seq
# Input is the seq length
calc_max_pred()
{
  echo $(python3 -c "import math; print(math.ceil($1*$LM_PROB))")
}

# Configurations of the training
# CONFIGS is in the format of SeqLen,BatchSize,Steps,Warmup
# There are two sets. This is correspond to the two pre=training stages.
# The first stage is usually with Seq128 to 90%.
# The second stage is usually with Seq512 from 90% to 100%
CONFIGS="128,9,1600,1000 128,9,1600,1000"

# Horovod (number of workers)
NP=24

# Data directory
DATA_DIR=/data/wikipedia

###################################
### Parameters for SQuAD Tuning ###
###################################

DO_SQUAD=1

SQUAD_DATA_DIR=/data/squad/1.1

SQUAD_BATCH=1
SQUAD_SEQ=128
SQUAD_EPOCH=1
SQUAD_LN_RATE=3e-5
SQUAD_STRIDE=128
SQUAD_WARMUP=0.1
