export MODEL_NAME=google-t5/t5-small 
export MODEL_TYPE="t5"
export INFERENCE_PRECISION="bfloat16"
export TP_SIZE=1
export PP_SIZE=1
export WORLD_SIZE=1
export MAX_BEAM_WIDTH=1
python convert_checkpoint.py --model_type ${MODEL_TYPE} \
                --model_dir $MODEL_NAME \
                --output_dir tmp/trt_models/${MODEL_NAME}/${INFERENCE_PRECISION} \
                --tp_size ${TP_SIZE} \
                --pp_size ${PP_SIZE} \
                --weight_data_type float16 \
                --dtype ${INFERENCE_PRECISION}