# Step 1. Data preprocessing
DATA_DIR=../pkl_data/sighan-finetune
PRETRAIN_MODEL=../bert_model/chinese-roberta-wwm-ext
mkdir -p $DATA_DIR


TRAIN_SRC_FILE=../data/sighan.train.ccl22.src
TRAIN_TRG_FILE=../data/sighan.train.ccl22.trg
DEV_SRC_FILE=../data/yaclc-csc-dev.src
DEV_TRG_FILE=../data/yaclc-csc-dev.lbl


if [ ! -f $DATA_DIR"/train.jsonl" ]; then
    python3 ./data_preprocess.py \
    --source_dir $TRAIN_SRC_FILE \
    --target_dir $TRAIN_TRG_FILE \
    --bert_path $PRETRAIN_MODEL \
    --save_path $DATA_DIR"/train.jsonl" \
    --data_mode "para" \
    --normalize "True"
fi

if [ ! -f $DATA_DIR"/dev.jsonl" ]; then
    python3 ./data_preprocess.py \
    --source_dir $DEV_SRC_FILE \
    --target_dir $DEV_TRG_FILE \
    --bert_path $PRETRAIN_MODEL \
    --save_path $DATA_DIR"/dev.jsonl" \
    --data_mode "lbl" \
    --normalize "True"
fi


# Step 2. Training
MODEL_DIR=../model/ratio1-finetune
BASE_DIR=../model/ratio1-finetune/roberta-r1-epoch-1.pt
CUDA_DEVICE=7
mkdir -p $MODEL_DIR/bak
cp finetune.sh $MODEL_DIR/bak
cp train.py $MODEL_DIR/bak

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 -u train.py \
    --pretrained_model $PRETRAIN_MODEL \
    --train_path $DATA_DIR"/train.jsonl" \
    --dev_path $DATA_DIR"/dev.jsonl" \
    --src_path $DEV_SRC_FILE \
    --lbl_path $DEV_TRG_FILE \
    --save_path $MODEL_DIR \
    --lr 3e-5 \
    --batch_size 32 \
    --accumulation_steps 1 \
    --tie_cls_weight True \
    --weight 2.\
    --tag "roberta-s" \
    --finetune_base $BASE_DIR\
    --seed 24\
    2>&1 | tee $MODEL_DIR"/roberta-s.log.txt"
