"""Implements the adapters' configurations."""

from collections import OrderedDict
from dataclasses import dataclass

import torch.nn as nn

@dataclass
class AdapterConfig(object):
    """Implements the adapter configuration proposed by Houlsby et. al, 2019
    in https://arxiv.org/abs/1902.00751."""
    add_layer_norm_before_adapter: bool = False
    add_layer_norm_after_adapter: bool = True
    non_linearity: str = "swish"
    task_reduction_factor: int = 16
    lang_reduction_factor: int = 16
    add_adapter_in_feed_forward = True
    add_adapter_in_self_attention = True
    # Whether to use conditional layer norms for adapters.
    conditional_layer_norm = False
    hidden_dim = 128
    lang_adapter_layers_encoder = None
    lang_adapter_layers_decoder = None
    task_adapter_layers_encoder = None
    task_adapter_layers_decoder = None
    lang_adapter_in_decoder = True
    task_adapter_in_decoder = True
    task_embedding_dim = 64
    task_embedding_dir = None
    parametric_task_embedding = False
    lang_embedding_dim = 32
    lang_embedding_dir = None
    parametric_lang_embedding = False
    task_hidden_dim = 128
    lang_hidden_dim = 128
    unique_hyper_net_layer_norm = True
    # If set, then it computes the adapters weights using a random matrix
    # multiplied by a vector which captures the intrinsic dimension of the
    # problem at hand.
    intrinsic_adapters = False
    intrinsic_dim = 100
    normalize_intrinsic_projections = False
    # This can be either random, or fastfood.
    intrinsic_projection = "random"
    keep_adapters_initial_values = True

    # Hypercomplex adapters parameters 
    hypercomplex_adapters = False
    hypercomplex_division = 8
    learn_phm = True
    normalize_phm_weight = False
    hypercomplex_nonlinearity="glorot-uniform"
    shared_phm_rule = False 
    factorized_phm = False 
    gradient_clip = False 
    phm_clamp = True 
    shared_W_phm = False 
    factorized_phm_rule = False 
    phm_c_init = "normal"
    phm_rank = 1
    phm_init_range = 0.01
    # If this is set, put the dropout at the end.
    dropout_at_end = False  
    prefix_dim = 100 
    init_prefix_from_vocab = False 
    kronecker_prod = False  
    bitfit = False 

 
class MetaAdapterConfig(AdapterConfig):
    """Implements Meta adapter in which a hyper-network generates the parameters of
     adapter layers. In this case we have a task embeddings which is feed to the
     hyper-network to allow it generate the weights for the adapter layers."""
    task_embedding_dim = 64
    task_embedding_dir = None
    parametric_task_embedding = False
    lang_embedding_dim = 32
    lang_embedding_dir = None 
    parametric_lang_embedding = False
    unique_hyper_net_layer_norm = True
    hidden_dim = 128
    task_hidden_dim = 128
    lang_hidden_dim = 128

ADAPTER_CONFIG_MAPPING = OrderedDict(
    [("adapter", AdapterConfig)
    ])


class AutoAdapterConfig(nn.Module):
    """Generic Adapter config class to instantiate different adapter configs."""

    @classmethod
    def get(cls, config_name: str):
        if config_name in ADAPTER_CONFIG_MAPPING:
            return ADAPTER_CONFIG_MAPPING[config_name]()
        raise ValueError(
            "Unrecognized adapter config type identifier: {}. Should contain one of {}"
                .format(config_name, ", ".join(ADAPTER_CONFIG_MAPPING.keys())))
