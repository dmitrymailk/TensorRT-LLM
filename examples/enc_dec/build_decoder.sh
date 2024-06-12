export MODEL_NAME=google-t5/t5-small 
export MODEL_TYPE="t5"
export INFERENCE_PRECISION="bfloat16"
export TP_SIZE=1
export PP_SIZE=1
export WORLD_SIZE=1
export MAX_BEAM_WIDTH=1

	# --gemm_plugin ${INFERENCE_PRECISION} \
# trtllm-build --checkpoint_dir tmp/trt_models/${MODEL_NAME}/${INFERENCE_PRECISION}/tp${TP_SIZE}/pp${PP_SIZE}/decoder \
# 	--output_dir tmp/trt_engines/${MODEL_NAME}/${WORLD_SIZE}-gpu/${INFERENCE_PRECISION}/tp${TP_SIZE}/decoder \
# 	--paged_kv_cache enable \
# 	--moe_plugin disable \
# 	--enable_xqa disable \
# 	--use_custom_all_reduce disable \
# 	--max_beam_width ${MAX_BEAM_WIDTH} \
# 	--max_batch_size 8 \
# 	--max_output_len 200 \
# 	--gemm_plugin auto \
# 	--bert_attention_plugin auto \
# 	--gpt_attention_plugin auto \
# 	--remove_input_padding enable \
# 	--context_fmha disable \
# 	--max_input_len 1

trtllm-build --checkpoint_dir tmp/trt_models/${MODEL_NAME}/${INFERENCE_PRECISION}/tp${TP_SIZE}/pp${PP_SIZE}/decoder \
            --output_dir tmp/trt_engines/${MODEL_NAME}/${WORLD_SIZE}-gpu/${INFERENCE_PRECISION}/tp${TP_SIZE}/decoder \
            --gemm_plugin auto \
            --strongly_typed