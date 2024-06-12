export MODEL_NAME=google-t5/t5-small 
export MODEL_TYPE="t5"
# export INFERENCE_PRECISION="bfloat16"
export INFERENCE_PRECISION="int4"
export TP_SIZE=1
export PP_SIZE=1
export WORLD_SIZE=1
export MAX_BEAM_WIDTH=1
# python convert_checkpoint_t5.py --model_type ${MODEL_TYPE} \
#                 --model_dir $MODEL_NAME \
#                 --output_dir tmp/trt_models/${MODEL_NAME}/${INFERENCE_PRECISION} \
#                 --tp_size ${TP_SIZE} \
#                 --pp_size ${PP_SIZE} \
#                 --weight_data_type float16 \
#                 --dtype ${INFERENCE_PRECISION} \
#                 --weight_only_precision
# python convert_checkpoint.py --model_type t5 \
python convert_checkpoint_t5.py --model_type t5 \
                --model_dir $MODEL_NAME \
                --output_dir tmp/trt_models/google-t5/t5-small/bfloat16 \
                --weight_data_type float16 \
                --dtype bfloat16 \
                --weight_only_precision

# bash build_decoder.sh
# bash build_encoder.sh
# bash test_t5.sh