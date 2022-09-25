# Step 1. Data preprocessing
DATA_DIR=../pkl_data/4600w-ratio1
BERT_MODEL=../bert_model/chinese-roberta-wwm-ext
mkdir -p $DATA_DIR


TRAIN_SRC_FILE=../data/4600w-ratio1.src
TRAIN_TRG_FILE=../data/4600w-ratio1.trg
DEV_SRC_FILE=../data/yaclc-csc-dev.src
DEV_TRG_FILE=../data/yaclc-csc-dev.lbl


if [ ! -f $DATA_DIR"/train.jsonl" ]; then
    python3 ./data_preprocess.py \
    --source_dir $TRAIN_SRC_FILE \
    --target_dir $TRAIN_TRG_FILE \
    --bert_path $BERT_MODEL \
    --save_path $DATA_DIR"/train.jsonl" \
    --data_mode "para" \
    --normalize "True"
fi

if [ ! -f $DATA_DIR"/dev.jsonl" ]; then
    python3 ./data_preprocess.py \
    --source_dir $DEV_SRC_FILE \
    --target_dir $DEV_TRG_FILE \
    --bert_path $BERT_MODEL \
    --save_path $DATA_DIR"/dev.jsonl" \
    --data_mode "lbl" \
    --normalize "True"
fi


# Step 2. Training
MODEL_DIR=../model/4600w-ratio1
CUDA_DEVICE=5
mkdir -p $MODEL_DIR/bak
cp pipeline.sh $MODEL_DIR/bak
cp train.py $MODEL_DIR/bak

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 -u code/train.py \
    --pretrained_model $BERT_MODEL \
    --train_path $DATA_DIR"/train.jsonl" \
    --dev_path $DATA_DIR"/dev.jsonl" \
    --src_path $DEV_SRC_FILE \
    --lbl_path $DEV_TRG_FILE \
    --save_path $MODEL_DIR \
    --batch_size 128 \
    --weight 5. \
    --lr 2e-4 \
    --accumulation_steps 2 \
    --evaluation_steps 100000 \
    --tie_cls_weight True \
    --tag "roberta-r1" \
    --seed 15 \
    2>&1 | tee $MODEL_DIR"/roberta-r1.log.txt"
