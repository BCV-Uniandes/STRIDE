DATA_PATH="data"
TRAIN_FOLD="fold2"
TEST_FOLD="fold1"
CONFIG_PATH="config/STRIDE/STRIDE_4scale.py"
PRETRAINED_PATH="pretrained_models/"
PRETRAINED_MODEL=$PRETRAINED_PATH"stride_"$TRAIN_FOLD".pth"
NUM_GPUS=6

OUTPUT_DIR="outputs/train-"$TRAIN_FOLD"_test"$TEST_FOLD

CUDA_VISIBLE_DEVICES=0,1,2,3,5,7 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS main.py \
	--output_dir $OUTPUT_DIR -c $CONFIG_PATH --coco_path $DATA_PATH \
	--train_fold $TRAIN_FOLD --test_fold $TEST_FOLD \
	--pretrain_model_path $PRETRAINED_MODEL \
	--options freeze=False regression=False