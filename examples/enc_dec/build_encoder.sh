export MODEL_NAME=google-t5/t5-small 
export MODEL_TYPE="t5"
export INFERENCE_PRECISION="bfloat16"
export TP_SIZE=1
export PP_SIZE=1
export WORLD_SIZE=1
export MAX_BEAM_WIDTH=1

# trtllm-build --checkpoint_dir tmp/trt_models/${MODEL_NAME}/${INFERENCE_PRECISION}/tp${TP_SIZE}/pp${PP_SIZE}/encoder \
# 	--output_dir tmp/trt_engines/${MODEL_NAME}/${WORLD_SIZE}-gpu/${INFERENCE_PRECISION}/tp${TP_SIZE}/encoder \
# 	--paged_kv_cache disable \
# 	--moe_plugin disable \
# 	--enable_xqa disable \
# 	--use_custom_all_reduce disable \
# 	--max_beam_width ${MAX_BEAM_WIDTH} \
# 	--max_batch_size 8 \
# 	--max_output_len 200 \
# 	--gemm_plugin ${INFERENCE_PRECISION} \
# 	--bert_attention_plugin ${INFERENCE_PRECISION} \
# 	--gpt_attention_plugin ${INFERENCE_PRECISION} \
# 	--remove_input_padding enable \
# 	--context_fmha disable

trtllm-build --checkpoint_dir tmp/trt_models/${MODEL_NAME}/${INFERENCE_PRECISION}/tp${TP_SIZE}/pp${PP_SIZE}/encoder \
            --output_dir tmp/trt_engines/${MODEL_NAME}/${WORLD_SIZE}-gpu/${INFERENCE_PRECISION}/tp${TP_SIZE}/encoder 
            # --gemm_plugin auto \
            # --strongly_typed
# trtllm-build --checkpoint_dir tmp/trt_models/google-t5/t5-small/bfloat16/tp1/pp1/encoder \
#             --output_dir tmp/trt_engines/google-t5/t5-small/1-gpu/bfloat16/tp1/encoder \
#             --gemm_plugin auto \
#             --strongly_typed