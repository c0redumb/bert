#!/bin/bash

#export NCCL_P2P_LEVEL=4
#export HSA_FORCE_FINE_GRAIN_PCIE=1
#export NCCL_MIN_NRINGS=4
#export NCCL_DEBUG=INFO

SCRIPTPATH=$(dirname $(realpath $0))

# Source parameters
source $SCRIPTPATH/training_cluster_params.sh

LAST_STAGE=
find_last_stage()
{
  DIRLIST="$1/Pretrain_Stage*"
  if ls $DIRLIST 1> /dev/null 2>&1; then
    echo "Trying to find the latest training stage"
  else
    echo "No pretraining stages found"
    return
  fi

  LAST=-1
  for dir in $DIRLIST; do
    #echo $dir
    if [ ! -d $dir ]; then
      continue
    fi
    NUM=$(echo $dir | rev | cut -d '_' -f 1 | rev)
    #echo $NUM
    if [ "$LAST" -lt "$NUM" ]; then
        LAST=$NUM
        LAST_STAGE=$dir
    fi
  done

  echo LAST_STAGE is $LAST_STAGE
}

SQUAD_DIR=$TRAIN_DIR/Squad_Training
rm -rf $SQUAD_DIR
mkdir -p $SQUAD_DIR

find_last_stage $TRAIN_DIR
#echo Last stage of pretraining is at $LAST_STAGE > $SQUAD_DIR/run_output.txt

echo "Model        : $MODEL"                     > $SQUAD_DIR/run_record.txt
echo "Epoch/Batch  : $SQUAD_EPOCH/$SQUAD_BATCH" >> $SQUAD_DIR/run_record.txt
echo "Seq/Stride   : $SQUAD_SEQ/$SQUAD_STRIDE"  >> $SQUAD_DIR/run_record.txt
echo "Warmup       : $SQUAD_WARMUP"             >> $SQUAD_DIR/run_record.txt
echo "Learn Rate   : $SQUAD_LN_RATE"            >> $SQUAD_DIR/run_record.txt
echo "Init Ckpt    : $LAST_STAGE"               >> $SQUAD_DIR/run_record.txt
echo "STARTED on " $(date)                      >> $SQUAD_DIR/run_record.txt

# run pretraining     -x HOROVOD_AUTOTUNE=1 \
  #   -x NCCL_P2P_LEVEL=4 \
  # -x NCCL_SOCKET_IFNAME=ib \
# mpirun --allow-run-as-root -np $NP \
#   -hostfile /data/run/hostfile \
#   -bind-to none -map-by slot \
#   -x NCCL_DEBUG=INFO \
#   -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
#   -x LD_LIBRARY_PATH -x PATH \
#   -mca pml ob1 -mca btl ^openib \
python3 $CODE_DIR/run_squad.py \
  --vocab_file=$TRAIN_DIR/vocab.txt \
  --bert_config_file=$TRAIN_DIR/bert_config.json \
  --init_checkpoint=$LAST_STAGE \
  --do_train=True \
  --train_file=$SQUAD_DATA_DIR/train-v1.1.json \
  --do_predict=True \
  --predict_file=$SQUAD_DATA_DIR/dev-v1.1.json \
  --train_batch_size=$SQUAD_BATCH \
  --learning_rate=$SQUAD_LN_RATE \
  --num_train_epochs=$SQUAD_EPOCH \
  --max_seq_length=$SQUAD_SEQ \
  --doc_stride=$SQUAD_STRIDE \
  --output_dir=$SQUAD_DIR \
2>&1 | tee $SQUAD_DIR/run_output.txt

echo "Run time     :" $SECONDS sec >> $SQUAD_DIR/run_record.txt
echo "times output :"              >> $SQUAD_DIR/run_record.txt
times                              >> $SQUAD_DIR/run_record.txt
echo "FINISHED on " $(date)        >> $SQUAD_DIR/run_record.txt

