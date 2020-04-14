#!/bin/bash

SCRIPTPATH=$(dirname $(realpath $0))

# Source parameters
source $SCRIPTPATH/training_cluster_params.sh

# Run pretraining
$SCRIPTPATH/training_cluster_pretrain.sh

# Run finetuning (SQuAD)
if [ ! -z $DO_SQUAD ]; then
  $SCRIPTPATH/training_cluster_squad.sh
fi

