# Inferencing w/ single GPU greedy search, compare results with HuggingFace FP32
export MODEL_NAME=google-t5/t5-small 
export MODEL_TYPE="t5"
export INFERENCE_PRECISION="bfloat16"
export TP_SIZE=1
export PP_SIZE=1
export WORLD_SIZE=1
export MAX_BEAM_WIDTH=1

python3 run.py --engine_dir tmp/trt_engines/${MODEL_NAME}/${WORLD_SIZE}-gpu/${INFERENCE_PRECISION}/tp${TP_SIZE} \
	--engine_name ${MODEL_NAME} \
	--model_name $MODEL_NAME \
	--max_new_token=64 \
	--num_beams=1 \
	--compare_hf_fp32