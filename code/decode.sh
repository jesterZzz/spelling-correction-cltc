PRETRAIN_MODEL=../bert_model/bert-base-chinese
DATA_DIR=../data/try

TEST_SRC_FILE=../data/yaclc-csc-test.src
TAG=report

python3 ./data_preprocess.py \
--source_dir $TEST_SRC_FILE \
--target_dir $TEST_TRG_FILE \
--is_test True \
--bert_path $PRETRAIN_MODEL \
--save_path $DATA_DIR"/test_"$TAG".jsonl" \
--data_mode "lbl" \
--normalize "True"

MODEL_PATH="(''，'')"
SAVE_PATH=../model/try

mkdir -p $SAVE_PATH

CUDA_VISIBLE_DEVICES=5 python3 decode.py \
	--pretrained_model $PRETRAIN_MODEL \
	--test_path $DATA_DIR"/test_"$TAG".jsonl" \
	--model_path $MODEL_PATH \
	--save_path $SAVE_PATH"/"$TAG".eval" \
	--batch_size 32 \
	--mode test;


