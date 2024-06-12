import argparse
import configparser
import copy
import json
import logging
import os
import types
from ast import literal_eval
from datetime import datetime
from pathlib import Path

import safetensors
from helper import convert_weight_to_dtype, fuse_qkv_one_layer, reshape, split
from transformers import (
    AutoModelForSeq2SeqLM,
    MBartForConditionalGeneration,
    Pix2StructForConditionalGeneration,
    T5ForConditionalGeneration,
    VisionEncoderDecoderModel,
)

from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.functional import LayerNormPositionType, LayerNormType, MLPType
from tensorrt_llm.models import PretrainedConfig

dir_path = os.path.dirname(os.path.realpath(__file__))
LOGGER = logging.getLogger(__name__)

layernorm_type_map = {i.name: i.value for i in LayerNormType}
layernorm_position_map = {i.name: i.value for i in LayerNormPositionType}
mlp_type_map = {i.name: i.value for i in MLPType}


def copy_args_to_component_config(component_config, args):
    for arg in vars(args):
        setattr(component_config, arg, getattr(args, arg))
    return component_config


def parse_t5_config(args, hf_model):
    config = configparser.ConfigParser()

    config["encoder"] = {}
    for key, val in hf_model.encoder.config.to_dict().items():
        config["encoder"][key] = f"{val}"
    config["encoder"]["weight_data_type"] = args.weight_data_type

    # manually set q_scaling to offset attention scaling's effect.
    # TODO: modify kernels to control whether to disable attention scaling
    def get_offset_q_scaling(config):
        scaling = 1 / config.head_size**0.5
        return scaling

    config["decoder"] = {}
    for key, val in hf_model.decoder.config.to_dict().items():
        config["decoder"][key] = f"{val}"
    config["decoder"]["weight_data_type"] = args.weight_data_type

    config["structure"] = dict()
    config["structure"]["t5_with_bias"] = "false"
    config["structure"]["use_gated_activation"] = str(
        hf_model.encoder.config.is_gated_act
    )
    config["structure"]["position_embedding_type"] = "relative"
    config["structure"]["model_type"] = args.model_type

    def parse_t5_config_by_component(config, component, args):
        component_config = types.SimpleNamespace()
        component_config = copy_args_to_component_config(component_config, args)
        component_config.n_head = config.getint(component, "num_heads")
        component_config.head_size = config.getint(component, "d_kv")
        component_config.hidden_size = config.getint(component, "d_model")
        component_config.ffn_hidden_size = config.getint(component, "d_ff")
        component_config.vocab_size = config.getint(component, "vocab_size")
        component_config.n_positions = config.getint(
            component, "n_positions", fallback=512
        )
        component_config.has_position_embedding = config.getboolean(
            component, "has_position_embedding", fallback=False
        )  # TODO: hardcoded here

        component_config.has_token_type_embedding = config.getboolean(
            component, "has_token_type_embedding", fallback=False
        )
        component_config.has_embedding_layernorm = config.getboolean(
            component, "has_embedding_layernorm", fallback=False
        )
        component_config.has_embedding_scale = config.getboolean(
            component, "has_embedding_scale", fallback=False
        )
        component_config.q_scaling = get_offset_q_scaling(component_config)
        component_config.has_attention_qkvo_bias = config.getboolean(
            component, "has_attention_qkvo_bias", fallback=False
        )  # TODO: hardcoded here
        component_config.has_mlp_bias = config.getboolean(
            component, "has_mlp_bias", fallback=False
        )
        component_config.has_model_final_layernorm = config.getboolean(
            component, "has_model_final_layernorm", fallback=True
        )
        component_config.layernorm_eps = config.getfloat(
            component, "layer_norm_epsilon"
        )
        component_config.layernorm_position = layernorm_position_map[
            config.get(component, "layernorm_position", fallback="pre_layernorm")
        ]  # TODO: hardcoded here
        component_config.layernorm_type = layernorm_type_map[
            config.get(component, "layernorm_type", fallback="RmsNorm")
        ]
        component_config.hidden_act = config.get(component, "dense_act_fn")
        component_config.gated_act = config.getboolean(component, "is_gated_act")
        component_config.mlp_type = mlp_type_map[
            "GatedMLP" if component_config.gated_act else "MLP"
        ]
        component_config.num_buckets = config.getint(
            component, "relative_attention_num_buckets"
        )
        component_config.max_distance = config.getint(
            component, "relative_attention_max_distance"
        )
        component_config.position_embedding_type = config.get(
            "structure", "position_embedding_type"
        )
        component_config.logits_dtype = config.get(
            component, "logits_dtype", fallback="float32"
        )
        component_config.ckpt_weight_dtype = config.get(component, "weight_data_type")

        if component == "encoder":
            component_config.n_layer = config.getint(component, "num_layers")

            component_config.relative_attention = (
                config.get("structure", "position_embedding_type") == "relative"
            )

            component_config.ckpt_weight_dtype = config.get(
                component, "weight_data_type"
            )

        elif component == "decoder":
            component_config.n_layer = config.getint(component, "num_decoder_layers")
            component_config.has_lm_head_bias = config.getboolean(
                component, "has_lm_head_bias", fallback=False  # TODO: T5 with bias
            )
            component_config.relative_attention = config.getboolean(
                component, "relative_attention", fallback=True
            )
            component_config.rescale_before_lm_head = config.getboolean(
                component, "tie_word_embeddings"
            )  # default is True (for T5), but False for Flan-T5
            component_config.encoder_hidden_size = config.getint("encoder", "d_model")
            component_config.encoder_num_heads = config.getint("encoder", "num_heads")
            component_config.encoder_head_size = config.getint("encoder", "d_kv")

        else:
            assert False, "Unsupported component!"

        return component_config

    encoder_config = parse_t5_config_by_component(config, "encoder", args)
    decoder_config = parse_t5_config_by_component(config, "decoder", args)

    return encoder_config, decoder_config


def convert_t5_weights_to_tllm_safetensors_(config, component, params):
    weights = {}

    mapping = config.mapping

    convert_weight_to_dtype(params, config.dtype)
    hidden_size = config.hidden_size
    ffn_hidden_size = config.ffn_hidden_size
    num_layers = config.num_hidden_layers
    n_head = config.num_attention_heads
    head_size = config.head_size
    attention_hidden_size = (
        n_head * head_size
    )  # head size * num_heads not necessarily equals hidden_dim, such as Flan-T5

    hf_param_prefix = f"{component}"
    trtllm_layer_name = f"{component}_layers"
    trtllm_attn_layer_name = "attention" if component == "encoder" else "self_attention"
    trtllm_attn_layernorm_name = (
        "self_attention_layernorm" if component == "decoder" else "attention_layernorm"
    )
    hf_component_idx = 1 if component == "encoder" else 2

    def get_attn_module_name(component, block, layer, attn_type):
        return f"{component}.block.{int(block)}.layer.{int(layer)}.{attn_type}"

    weights["embedding.vocab_embedding.weight"] = reshape(
        params["shared.weight"].clone(), None
    )

    layers_range = mapping.pp_layers(num_layers)
    for layer_idx in layers_range:
        local_layer_idx = layer_idx - layers_range[0]
        trtllm_layer_name_prefix = f"{trtllm_layer_name}.{local_layer_idx}"
        hf_layer_name_prefix = f"{hf_param_prefix}.block.{layer_idx}"

        hidden_layer_name_split = {
            f"{hf_layer_name_prefix}.layer.0.SelfAttention.o.weight": {
                "name": f"{trtllm_layer_name_prefix}.{trtllm_attn_layer_name}.dense.weight",
                "shape": (hidden_size, attention_hidden_size // mapping.tp_size),
                "split_dim": -1,
            },
            f"{hf_layer_name_prefix}.layer.{hf_component_idx}.DenseReluDense.wo.weight": {
                "name": f"{trtllm_layer_name_prefix}.mlp.proj.weight",
                "shape": (hidden_size, ffn_hidden_size // mapping.tp_size),
                "split_dim": -1,
            },
            f"{hf_layer_name_prefix}.layer.{hf_component_idx}.DenseReluDense.wi.weight": {
                "name": f"{trtllm_layer_name_prefix}.mlp.fc.weight",
                "shape": (ffn_hidden_size // mapping.tp_size, hidden_size),
                "split_dim": 0,
            },
            f"{hf_layer_name_prefix}.layer.{hf_component_idx}.DenseReluDense.wi_0.weight": {
                "name": f"{trtllm_layer_name_prefix}.mlp.fc.weight",
                "shape": (ffn_hidden_size // mapping.tp_size, hidden_size),
                "split_dim": 0,
            },
        }

        hidden_layer_name_no_split = {
            f"{hf_layer_name_prefix}.layer.0.layer_norm.weight": {
                "name": f"{trtllm_layer_name_prefix}.{trtllm_attn_layernorm_name}.weight",
                "shape": None,
            },
            f"{hf_layer_name_prefix}.layer.{hf_component_idx}.layer_norm.weight": {
                "name": f"{trtllm_layer_name_prefix}.mlp_layernorm.weight",
                "shape": None,
            },
        }

        if config.gated_act:
            hidden_layer_name_split.update(
                {
                    f"{hf_layer_name_prefix}.layer.{hf_component_idx}.DenseReluDense.wi2.weight": {
                        "name": f"{trtllm_layer_name_prefix}.mlp.gate.weight",
                        "shape": (ffn_hidden_size // mapping.tp_size, hidden_size),
                        "split_dim": 0,
                    },
                    f"{hf_layer_name_prefix}.layer.{hf_component_idx}.DenseReluDense.wi_1.weight": {
                        "name": f"{trtllm_layer_name_prefix}.mlp.gate.weight",
                        "shape": (ffn_hidden_size // mapping.tp_size, hidden_size),
                        "split_dim": 0,
                    },
                }
            )

        if component == "decoder":
            hidden_layer_name_split.update(
                {
                    f"{hf_layer_name_prefix}.layer.1.EncDecAttention.o.weight": {
                        "name": f"{trtllm_layer_name_prefix}.cross_attention.dense.weight",
                        "shape": (
                            hidden_size,
                            attention_hidden_size // mapping.tp_size,
                        ),
                        "split_dim": -1,
                    },
                }
            )
            hidden_layer_name_no_split.update(
                {
                    f"{hf_layer_name_prefix}.layer.1.layer_norm.weight": {
                        "name": f"{trtllm_layer_name_prefix}.cross_attention_layernorm.weight",
                        "shape": None,
                    },
                }
            )
            self_attn_module_name = get_attn_module_name(
                component, layer_idx, "1", "EncDecAttention"
            )
            weights.update(
                fuse_qkv_one_layer(
                    params,
                    self_attn_module_name,
                    f"{trtllm_layer_name_prefix}.cross_attention",
                    mapping.tp_size,
                    mapping.tp_rank,
                    config.model_type,
                    (attention_hidden_size * 3 // mapping.tp_size, hidden_size),
                    None,
                )
            )

        self_attn_module_name = get_attn_module_name(
            component, layer_idx, "0", "SelfAttention"
        )
        weights.update(
            fuse_qkv_one_layer(
                params,
                self_attn_module_name,
                f"{trtllm_layer_name_prefix}.{trtllm_attn_layer_name}",
                mapping.tp_size,
                mapping.tp_rank,
                config.model_type,
                (attention_hidden_size * 3 // mapping.tp_size, hidden_size),
                None,
            )
        )

        weights[
            f"{trtllm_layer_name_prefix}.{trtllm_attn_layer_name}.rel_attn_table"
        ] = reshape(
            split(
                params[
                    f"{component}.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
                ].T,
                mapping.tp_size,
                mapping.tp_rank,
                0,
            ),
            (n_head // mapping.tp_size, config.num_buckets),
        )

        for hf_weight_name, weight_info in hidden_layer_name_split.items():
            if hf_weight_name in params.keys():
                weights[weight_info["name"]] = reshape(
                    split(
                        params[hf_weight_name],
                        mapping.tp_size,
                        mapping.tp_rank,
                        dim=weight_info["split_dim"],
                    ),
                    weight_info["shape"],
                )
        for hf_weight_name, weight_info in hidden_layer_name_no_split.items():
            if hf_weight_name in params.keys():
                weights[weight_info["name"]] = reshape(
                    params[hf_weight_name].clone(), shape=weight_info["shape"]
                )

    weights[f"final_layernorm.weight"] = reshape(
        params[f"{component}.final_layer_norm.weight"].clone(), None
    )

    if component == "decoder":
        weights["lm_head.weight"] = reshape(
            split(params["lm_head.weight"], mapping.tp_size, mapping.tp_rank, dim=0),
            (config.vocab_size // mapping.tp_size, hidden_size),
        )

    return weights


def convert_t5_weights_to_tllm_safetensors(config, component, params):
    weights = {}

    mapping = config.mapping

    convert_weight_to_dtype(params, config.dtype)
    hidden_size = config.hidden_size
    ffn_hidden_size = config.ffn_hidden_size
    num_layers = config.num_hidden_layers
    n_head = config.num_attention_heads
    head_size = config.head_size
    attention_hidden_size = (
        n_head * head_size
    )  # head size * num_heads not necessarily equals hidden_dim, such as Flan-T5

    hf_param_prefix = f"{component}"
    trtllm_layer_name = f"{component}_layers"
    trtllm_attn_layer_name = "attention" if component == "encoder" else "self_attention"
    trtllm_attn_layernorm_name = (
        "self_attention_layernorm" if component == "decoder" else "attention_layernorm"
    )
    hf_component_idx = 1 if component == "encoder" else 2

    def get_attn_module_name(component, block, layer, attn_type):
        return f"{component}.block.{int(block)}.layer.{int(layer)}.{attn_type}"

    weights["embedding.vocab_embedding.weight"] = reshape(
        params["shared.weight"].clone(), None
    )

    # {
    #     "encoder_layers.4.attention.qkv.weights_scaling_factor",
    #     "encoder_layers.5.attention.qkv.prequant_scaling_factor",
    #     "encoder_layers.2.attention.qkv.prequant_scaling_factor",
    #     "encoder_layers.1.attention.qkv.prequant_scaling_factor",
    #     "encoder_layers.4.attention.qkv.prequant_scaling_factor",
    #     "encoder_layers.5.attention.qkv.weights_scaling_factor",
    #     "encoder_layers.2.attention.qkv.weights_scaling_factor",
    #     "encoder_layers.0.attention.qkv.weights_scaling_factor",
    #     "encoder_layers.3.attention.qkv.prequant_scaling_factor",
    #     "encoder_layers.3.attention.qkv.weights_scaling_factor",
    #     "encoder_layers.1.attention.qkv.weights_scaling_factor",
    #     "encoder_layers.0.attention.qkv.prequant_scaling_factor",
    # }
    layers_range = mapping.pp_layers(num_layers)
    for layer_idx in layers_range:
        local_layer_idx = layer_idx - layers_range[0]
        trtllm_layer_name_prefix = f"{trtllm_layer_name}.{local_layer_idx}"
        hf_layer_name_prefix = f"{hf_param_prefix}.block.{layer_idx}"

        hidden_layer_name_split = {
            f"{hf_layer_name_prefix}.layer.0.SelfAttention.o.weight": {
                "name": f"{trtllm_layer_name_prefix}.{trtllm_attn_layer_name}.dense.weight",
                "shape": (hidden_size, attention_hidden_size // mapping.tp_size),
                "split_dim": -1,
            },
            f"{hf_layer_name_prefix}.layer.{hf_component_idx}.DenseReluDense.wo.weight": {
                "name": f"{trtllm_layer_name_prefix}.mlp.proj.weight",
                "shape": (hidden_size, ffn_hidden_size // mapping.tp_size),
                "split_dim": -1,
            },
            f"{hf_layer_name_prefix}.layer.{hf_component_idx}.DenseReluDense.wi.weight": {
                "name": f"{trtllm_layer_name_prefix}.mlp.fc.weight",
                "shape": (ffn_hidden_size // mapping.tp_size, hidden_size),
                "split_dim": 0,
            },
            f"{hf_layer_name_prefix}.layer.{hf_component_idx}.DenseReluDense.wi_0.weight": {
                "name": f"{trtllm_layer_name_prefix}.mlp.fc.weight",
                "shape": (ffn_hidden_size // mapping.tp_size, hidden_size),
                "split_dim": 0,
            },
        }

        hidden_layer_name_no_split = {
            f"{hf_layer_name_prefix}.layer.0.layer_norm.weight": {
                "name": f"{trtllm_layer_name_prefix}.{trtllm_attn_layernorm_name}.weight",
                "shape": None,
            },
            f"{hf_layer_name_prefix}.layer.{hf_component_idx}.layer_norm.weight": {
                "name": f"{trtllm_layer_name_prefix}.mlp_layernorm.weight",
                "shape": None,
            },
            f"{hf_layer_name_prefix}.layer.0.SelfAttention.o.input_quantizer._pre_quant_scale": {
                "name": f"{trtllm_layer_name_prefix}.{trtllm_attn_layer_name}.dense.prequant_scaling_factor",
                "shape": (hidden_size),
                "split_dim": None,
            },
            f"{hf_layer_name_prefix}.layer.0.SelfAttention.o.weight_quantizer._amax": {
                "name": f"{trtllm_layer_name_prefix}.{trtllm_attn_layer_name}.dense.weights_scaling_factor",
                "shape": tuple(
                    params[
                        f"{hf_layer_name_prefix}.layer.0.SelfAttention.o.weight_quantizer._amax"
                    ].shape
                ),
                "split_dim": None,
            },
            f"{hf_layer_name_prefix}.layer.{hf_component_idx}.DenseReluDense.wo.input_quantizer._pre_quant_scale": {
                "name": f"{trtllm_layer_name_prefix}.mlp.proj.prequant_scaling_factor",
                "shape": (ffn_hidden_size),
                "split_dim": None,
            },
            f"{hf_layer_name_prefix}.layer.{hf_component_idx}.DenseReluDense.wo.weight_quantizer._amax": {
                "name": f"{trtllm_layer_name_prefix}.mlp.proj.weights_scaling_factor",
                "shape": tuple(
                    params[
                        f"{hf_layer_name_prefix}.layer.{hf_component_idx}.DenseReluDense.wo.weight_quantizer._amax"
                    ].shape
                ),
                "split_dim": None,
            },
            f"{hf_layer_name_prefix}.layer.{hf_component_idx}.DenseReluDense.wi.input_quantizer._pre_quant_scale": {
                "name": f"{trtllm_layer_name_prefix}.mlp.fc.prequant_scaling_factor",
                "shape": (hidden_size),
                "split_dim": None,
            },
            f"{hf_layer_name_prefix}.layer.{hf_component_idx}.DenseReluDense.wi.weight_quantizer._amax": {
                "name": f"{trtllm_layer_name_prefix}.mlp.fc.weights_scaling_factor",
                # "shape": (hidden_size),
                "shape": tuple(
                    params[
                        f"{hf_layer_name_prefix}.layer.{hf_component_idx}.DenseReluDense.wi.weight_quantizer._amax"
                    ].shape
                ),
                "split_dim": None,
            },
            
        }

        if config.gated_act:
            hidden_layer_name_split.update(
                {
                    f"{hf_layer_name_prefix}.layer.{hf_component_idx}.DenseReluDense.wi2.weight": {
                        "name": f"{trtllm_layer_name_prefix}.mlp.gate.weight",
                        "shape": (ffn_hidden_size // mapping.tp_size, hidden_size),
                        "split_dim": 0,
                    },
                    f"{hf_layer_name_prefix}.layer.{hf_component_idx}.DenseReluDense.wi_1.weight": {
                        "name": f"{trtllm_layer_name_prefix}.mlp.gate.weight",
                        "shape": (ffn_hidden_size // mapping.tp_size, hidden_size),
                        "split_dim": 0,
                    },
                }
            )

        if component == "decoder":
            hidden_layer_name_split.update(
                {
                    f"{hf_layer_name_prefix}.layer.1.EncDecAttention.o.weight": {
                        "name": f"{trtllm_layer_name_prefix}.cross_attention.dense.weight",
                        "shape": (
                            hidden_size,
                            attention_hidden_size // mapping.tp_size,
                        ),
                        "split_dim": -1,
                    },
                }
            )
            hidden_layer_name_no_split.update(
                {
                    f"{hf_layer_name_prefix}.layer.1.layer_norm.weight": {
                        "name": f"{trtllm_layer_name_prefix}.cross_attention_layernorm.weight",
                        "shape": None,
                    },
                }
            )
            self_attn_module_name = get_attn_module_name(
                component, layer_idx, "1", "EncDecAttention"
            )
            weights.update(
                fuse_qkv_one_layer(
                    params,
                    self_attn_module_name,
                    f"{trtllm_layer_name_prefix}.cross_attention",
                    mapping.tp_size,
                    mapping.tp_rank,
                    config.model_type,
                    (attention_hidden_size * 3 // mapping.tp_size, hidden_size),
                    None,
                )
            )

        self_attn_module_name = get_attn_module_name(
            component, layer_idx, "0", "SelfAttention"
        )
        weights.update(
            fuse_qkv_one_layer(
                params,
                self_attn_module_name,
                f"{trtllm_layer_name_prefix}.{trtllm_attn_layer_name}",
                mapping.tp_size,
                mapping.tp_rank,
                config.model_type,
                (attention_hidden_size * 3 // mapping.tp_size, hidden_size),
                None,
            )
        )

        weights[
            f"{trtllm_layer_name_prefix}.{trtllm_attn_layer_name}.rel_attn_table"
        ] = reshape(
            split(
                params[
                    f"{component}.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
                ].T,
                mapping.tp_size,
                mapping.tp_rank,
                0,
            ),
            (n_head // mapping.tp_size, config.num_buckets),
        )

        for hf_weight_name, weight_info in hidden_layer_name_split.items():
            if hf_weight_name in params.keys():
                weights[weight_info["name"]] = reshape(
                    split(
                        params[hf_weight_name],
                        mapping.tp_size,
                        mapping.tp_rank,
                        dim=weight_info["split_dim"],
                    ),
                    weight_info["shape"],
                )
        for hf_weight_name, weight_info in hidden_layer_name_no_split.items():
            if hf_weight_name in params.keys():
                weights[weight_info["name"]] = reshape(
                    params[hf_weight_name].clone(), shape=weight_info["shape"]
                )

    weights[f"final_layernorm.weight"] = reshape(
        params[f"{component}.final_layer_norm.weight"].clone(), None
    )

    if component == "decoder":
        weights["lm_head.weight"] = reshape(
            split(params["lm_head.weight"], mapping.tp_size, mapping.tp_rank, dim=0),
            (config.vocab_size // mapping.tp_size, hidden_size),
        )

    return weights


def get_model(args):
    if args.model_type == "t5":
        print("args.model_dir", args.model_dir)
        model = T5ForConditionalGeneration.from_pretrained(args.model_dir)
    elif args.model_type == "nmt":
        from fairseq.models.transformer import TransformerModel

        model = TransformerModel.from_pretrained(args.model_dir)
    elif args.model_type == "bart":
        if args.nougat:
            model = VisionEncoderDecoderModel.from_pretrained(args.model_dir)
            model = model.get_decoder()
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    elif args.model_type == "pix2struct":
        model = Pix2StructForConditionalGeneration.from_pretrained(args.model_dir)
    return model


def quantize_model(model):
    import modelopt.torch.quantization as mtq
    import modelopt.torch.quantization as mtq
    from tensorrt_llm.models.llama.convert import load_calib_dataset
    from tqdm import tqdm
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import copy
    from datasets import load_dataset

    calib_dataset = "ccdv/cnn_dailymail"
    dataset = load_dataset(calib_dataset, "3.0.0")["train"]
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")

    def forward_loop(model):
        # for batch in data_loader:
        #     model(batch)
        for i in tqdm(range(5), desc="calibrating model"):
            datapoint = dataset[i]
            line = copy.copy(datapoint)
            # print(line)
            article = line["article"][:500] + " TL;DR: ".strip()
            article = article.strip().replace(" n't", "n't")

            highlights = line["highlights"][:500].strip().replace(" n't", "n't")
            input_ids = tokenizer(
                article,
                return_tensors="pt",
                max_length=256,
                padding=True,
                truncation=True,
            ).input_ids.to("cuda")
            labels = tokenizer(
                highlights,
                return_tensors="pt",
                max_length=256,
                padding=True,
                truncation=True,
            ).input_ids.to("cuda")

            model(
                input_ids=input_ids,
                labels=labels,
            )

    config = mtq.INT4_AWQ_CFG
    # Quantize the model and perform calibration (PTQ)
    model = mtq.quantize(model, config, forward_loop)
    return model


def convert_checkpoint(args):

    model = get_model(args)
    model.to("cuda")
    model = quantize_model(model=model)

    model = model.to(str_dtype_to_torch(args.weight_data_type))

    saved_dir = Path(args.output_dir) / f"tp{args.tp_size}" / f"pp{args.pp_size}"
    saved_dir.mkdir(parents=True, exist_ok=True)

    encoder_saved_dir = saved_dir / "encoder"
    encoder_saved_dir.mkdir(parents=True, exist_ok=True)
    decoder_saved_dir = saved_dir / "decoder"
    decoder_saved_dir.mkdir(parents=True, exist_ok=True)

    world_size = args.tp_size * args.pp_size

    kv_cache_quant_algo = None
    quant_algo = {
        "quant_algo": "W4A16_AWQ",
        "kv_cache_quant_algo": "INT8",
        "group_size": 128,
        "has_zero_point": False,
        "pre_quant_scale": True,
        "exclude_modules": ["lm_head"],
    }
    # quant_algo = {}

    encoder_config, decoder_config = globals()[f"parse_{args.model_type}_config"](
        args, model
    )

    additional_settings = ["gated_act"]

    if not args.nougat and args.model_type != "pix2struct":
        tllm_encoder_config = {
            "architecture": "EncoderModel",
            "dtype": args.dtype,
            "logits_dtype": encoder_config.logits_dtype,
            "num_hidden_layers": encoder_config.n_layer,
            "num_attention_heads": encoder_config.n_head,
            "hidden_size": encoder_config.hidden_size,
            "norm_epsilon": encoder_config.layernorm_eps,
            "vocab_size": encoder_config.vocab_size,
            "position_embedding_type": encoder_config.position_embedding_type,
            "hidden_act": encoder_config.hidden_act,
            "quantization": {
                **quant_algo
                # "quant_algo": quant_algo,
                # "kv_cache_quant_algo": kv_cache_quant_algo,
            },
            "mapping": {
                "world_size": world_size,
                "tp_size": args.tp_size,
                "pp_size": args.pp_size,
            },
            "use_parallel_embedding": args.use_parallel_embedding,
            "embedding_sharding_dim": args.embedding_sharding_dim,
            "share_embedding_table": args.use_embedding_sharing,
            "max_position_embeddings": encoder_config.n_positions,
            "num_key_value_heads": encoder_config.n_head,
            "use_prompt_tuning": args.max_prompt_embedding_table_size > 0,
            "head_size": encoder_config.head_size,
            "has_position_embedding": encoder_config.has_position_embedding,
            "layernorm_type": encoder_config.layernorm_type,
            "has_attention_qkvo_bias": encoder_config.has_attention_qkvo_bias,
            "has_mlp_bias": encoder_config.has_mlp_bias,
            "has_model_final_layernorm": encoder_config.has_model_final_layernorm,
            "has_embedding_layernorm": encoder_config.has_embedding_layernorm,
            "has_embedding_scale": encoder_config.has_embedding_scale,
            "ffn_hidden_size": encoder_config.ffn_hidden_size,
            "q_scaling": encoder_config.q_scaling,
            "layernorm_position": encoder_config.layernorm_position,
            "mlp_type": encoder_config.mlp_type,
            "relative_attention": encoder_config.relative_attention,
            "max_distance": encoder_config.max_distance,
            "num_buckets": encoder_config.num_buckets,
            "model_type": encoder_config.model_type,
        }

        for additional_setting in additional_settings:
            if hasattr(encoder_config, additional_setting):
                tllm_encoder_config.update(
                    {additional_setting: getattr(encoder_config, additional_setting)}
                )

        with (encoder_saved_dir / f"config.json").open("w") as f:
            json.dump(tllm_encoder_config, f, indent=4)

        encoder_convert_args = dict(params=model.state_dict(), component="encoder")
    tllm_decoder_config = {
        "architecture": "DecoderModel",
        "dtype": args.dtype,
        "logits_dtype": decoder_config.logits_dtype,
        "num_hidden_layers": decoder_config.n_layer,
        "num_attention_heads": decoder_config.n_head,
        "hidden_size": decoder_config.hidden_size,
        "norm_epsilon": decoder_config.layernorm_eps,
        "vocab_size": decoder_config.vocab_size,
        "position_embedding_type": decoder_config.position_embedding_type,
        "hidden_act": decoder_config.hidden_act,
        "quantization": {
            **quant_algo,
            # "quant_algo": quant_algo,
            # "kv_cache_quant_algo": kv_cache_quant_algo,
        },
        "mapping": {
            "world_size": world_size,
            "tp_size": args.tp_size,
            "pp_size": args.pp_size,
        },
        "use_parallel_embedding": args.use_parallel_embedding,
        "embedding_sharding_dim": args.embedding_sharding_dim,
        "share_embedding_table": args.use_embedding_sharing,
        "max_position_embeddings": decoder_config.n_positions,
        "use_prompt_tuning": args.max_prompt_embedding_table_size > 0,
        "head_size": decoder_config.head_size,
        "has_position_embedding": decoder_config.has_position_embedding,
        "layernorm_type": decoder_config.layernorm_type,
        "has_attention_qkvo_bias": decoder_config.has_attention_qkvo_bias,
        "has_mlp_bias": decoder_config.has_mlp_bias,
        "has_model_final_layernorm": decoder_config.has_model_final_layernorm,
        "has_embedding_layernorm": decoder_config.has_embedding_layernorm,
        "has_embedding_scale": decoder_config.has_embedding_scale,
        "ffn_hidden_size": decoder_config.ffn_hidden_size,
        "q_scaling": decoder_config.q_scaling,
        "layernorm_position": decoder_config.layernorm_position,
        "mlp_type": decoder_config.mlp_type,
        "relative_attention": decoder_config.relative_attention,
        "max_distance": decoder_config.max_distance,
        "num_buckets": decoder_config.num_buckets,
        "model_type": decoder_config.model_type,
        "rescale_before_lm_head": decoder_config.rescale_before_lm_head,
        "encoder_hidden_size": decoder_config.encoder_hidden_size,
        "encoder_num_heads": decoder_config.encoder_num_heads,
        "encoder_head_size": decoder_config.encoder_head_size,
        "skip_cross_qkv": args.skip_cross_qkv,
    }
    for additional_setting in additional_settings:
        if hasattr(decoder_config, additional_setting):
            tllm_decoder_config.update(
                {additional_setting: getattr(decoder_config, additional_setting)}
            )

    with (decoder_saved_dir / f"config.json").open("w") as f:
        json.dump(tllm_decoder_config, f, indent=4)

    decoder_convert_args = dict(params=model.state_dict(), component="decoder")

    if args.model_type == "nmt":
        fairseq_config = vars(model.cfg.model)  # Namespace --> dict
        num_embeddings = fairseq_config["max_source_positions"]
        embedding_dim = fairseq_config["encoder_embed_dim"]
        padding_idx = model.models[0].encoder.embed_tokens.padding_idx  # 1

        sin_pos_embedding = model.models[0].encoder.embed_positions.get_embedding(
            padding_idx + 1 + num_embeddings, embedding_dim, padding_idx=padding_idx
        )  # [2 + num_embeddings, embed_dim]
        sin_pos_embedding = sin_pos_embedding[2:, :]  # remove offset embeddings

        encoder_convert_args["sin_pos_embedding"] = sin_pos_embedding
        decoder_convert_args["sin_pos_embedding"] = sin_pos_embedding

    if args.workers == 1:
        if not args.nougat and args.model_type != "pix2struct":
            convert(
                0,
                world_size,
                args,
                tllm_encoder_config,
                encoder_convert_args,
                encoder_saved_dir,
            )
        convert(
            0,
            world_size,
            args,
            tllm_decoder_config,
            decoder_convert_args,
            decoder_saved_dir,
        )
    else:
        if args.workers > world_size:
            args.workers = world_size
        LOGGER.info(f"Convert checkpoint using {args.workers} workers.")
        import torch.multiprocessing as mp

        if not args.nougat and args.model_type != "pix2struct":
            mp.spawn(
                convert,
                nprocs=args.workers,
                args=(
                    world_size,
                    args,
                    tllm_encoder_config,
                    encoder_convert_args,
                    encoder_saved_dir,
                ),
            )
        mp.spawn(
            convert,
            nprocs=args.workers,
            args=(
                world_size,
                args,
                tllm_decoder_config,
                decoder_convert_args,
                decoder_saved_dir,
            ),
        )


def convert(worker_rank, world_size, args, model_config, convert_args, saved_dir):
    for rank in range(worker_rank, world_size, args.workers):
        rank_config = copy.deepcopy(PretrainedConfig.from_dict(model_config))
        rank_config.set_rank(rank)
        weights = globals()[
            f"convert_{rank_config.model_type}_weights_to_tllm_safetensors"
        ](config=rank_config, **convert_args)
        safetensors.torch.save_file(weights, f"{saved_dir}/rank{rank}.safetensors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--model_type",
        type=str,
        default="t5",
        choices=["t5", "nmt", "bart", "pix2struct"],
        help="Model to be converted.",
    )
    parser.add_argument(
        "--world_size", type=int, default=1, help="MPI world size (must equal TP * PP)"
    )
    parser.add_argument(
        "--tp_size", type=int, default=1, help="N-way tensor parallelism size"
    )
    parser.add_argument(
        "--pp_size", type=int, default=1, help="N-way pipeline parallelism size"
    )
    parser.add_argument(
        "--model_dir",
        "-i",
        type=str,
        help="Path to the framework checkpoint file",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        help="Path to the converted TRT-LLM model weight file",
        required=True,
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="How many workers to spawn for conversion (default: 4)",
        default=1,
    )
    parser.add_argument(
        "--weight_data_type",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )  # TODO: test support for bf16?
    parser.add_argument(
        "--nougat",
        action="store_true",
        help="Model which uses vision encoder + mbart decoder",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Provide verbose messages"
    )
    parser.add_argument(
        "--max_prompt_embedding_table_size",
        "--max_multimodal_len",
        type=int,
        default=0,
        help="Setting to a value > 0 enables support for prompt tuning or multimodal input.",
    )
    parser.add_argument(
        "--use_parallel_embedding",
        action="store_true",
        default=False,
        help="By default embedding parallelism is disabled. By setting this flag, embedding parallelism is enabled",
    )
    parser.add_argument(
        "--embedding_sharding_dim",
        type=int,
        default=0,
        choices=[0, 1],
        help="By default the embedding lookup table is sharded along vocab dimension (embedding_sharding_dim=0). "
        "To shard it along hidden dimension, set embedding_sharding_dim=1"
        "Note: embedding sharding is only enabled when embedding_sharding_dim = 0",
    )
    parser.add_argument(
        "--use_weight_only",
        default=False,
        action="store_true",
        help="Quantize weights for the various GEMMs to INT4/INT8."
        "See --weight_only_precision to set the precision",
    )
    parser.add_argument(
        "--weight_only_precision",
        const="int8",
        type=str,
        nargs="?",
        default="int8",
        choices=["int8", "int4"],
        help="Define the precision for the weights when using weight-only quantization."
        "You must also use --use_weight_only for that argument to have an impact.",
    )
    parser.add_argument(
        "--use_embedding_sharing",
        action="store_true",
        default=False,
        help="Try to reduce the engine size by sharing the embedding lookup table between two layers."
        "Note: the flag might not take effect when the criteria are not met.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Target inference dtype. Weights and Computation will be in this dtype, no matter what original dtype the weight checkpoint has.",
    )
    parser.add_argument(
        "--skip_cross_qkv",
        action="store_true",
        help="Skip redundant cross qkv computation by using TensorRT IfConditional switch (experimental).",
    )
    args = parser.parse_args()
    log_format = "%(asctime)s %(name)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format=log_format
    )
    LOGGER.info("\n=============== Argument ===============")
    for key in vars(args):
        LOGGER.info(f"{key}: {vars(args)[key]}")
    LOGGER.info("========================================")

    start_time = datetime.now()
    convert_checkpoint(args)
    stop_time = datetime.now()
    run_time = stop_time - start_time
    LOGGER.info("Spend {} (h:m:s) to convert the model".format(run_time))
