from seq2seq.adapters import ADAPTER_CONFIG_MAPPING
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class AdapterTrainingArguments:
    """Defines the adapters parameters."""
    freeze_lang_adapters: Optional[bool] = field(default=False, metadata={"help": "If set, freezes language adapters."})
    train_lang_adapters: Optional[bool] = field(default=False,
                                                metadata={"help": "If set, adds language adapters in the model."})
    train_task_adapters: Optional[bool] = field(default=False,
                                                metadata={"help": "If set, adds task adapters in the model."})
    adapter_config_name: Optional[str] = field(
        default="meta-adapter", metadata={"help": "config name for the adapter layers, should be selected "
        f"in {sorted(ADAPTER_CONFIG_MAPPING.keys())}."}
    )
    task_embedding_dir: Optional[str] = field(default=None,
        metadata={"help": "defines the directory containing task embeddings."
        "We expect each task have a file with its name there."})
    lang_embedding_dir: Optional[str] = field(default=None,
        metadata={"help": "defines the directory containing language embeddings."
        "We expect each language to have a file with its name there."})
    task_embedding_dim: Optional[int] = field(default=None, metadata={"help": "task embedding dimensions."})
    lang_embedding_dim: Optional[int] = field(default=None, metadata={"help": "language embedding dimensions."})
    add_layer_norm_before_adapter: Optional[bool] = field(default=False, metadata={
        "help": "whether to have layer-norm before adapter."})
    add_layer_norm_after_adapter: Optional[bool] = field(default=True,
        metadata={"help": "whether to have layer-norm after adapter."})
    hidden_dim: Optional[int] = field(default=128, metadata={"help": "defines the default hidden dimension for "
        "adapter layers."})
    lang_reduction_factor: Optional[int] = field(default=16, metadata={"help": "defines the default reduction factor for "
        "language adapter layers."})
    task_reduction_factor: Optional[int] = field(default=16, metadata={"help": "defines the default reduction factor for "
        "language adapter layers."})
    non_linearity: Optional[str] = field(default="swish", metadata={"help": "Defines nonlinearity for adapter layers."})
    conditional_layer_norm: Optional[bool] = field(default=False,
                                                   metadata={"help": "Implements conditional layer norms "
                                                                     "modulated based on task embeddings."})
    parametric_task_embedding: bool = field(default=False,
                                            metadata={"help": "If set, transform the task-embedding to a parameter "
                                                        "in case of meta-adapter."})
    parametric_lang_embedding: bool = field(default=False,
                                            metadata={"help": "If set, transform the lang-embedding to a parameter "
                                                        "in case of meta-adapter."})
    unfreeze_lm_head: bool = field(default=False, metadata={"help": "If set unfreeze the last linear layer."})
    unfreeze_layer_norms: bool = field(default=False, metadata={"help": "If set, unfreezes the layer norms."})
    lang_adapters_path: Optional[str] = field(default=None, metadata={"help": "Specifies the path to the trained"
                                                                              "language adapters."})
    lang_adapter_layers_encoder: Optional[List[int]] = field(default=None, metadata={"help": "Defines the layers id"
                                                                                      "in which language adapter is"
                                                                                      "added in the encoder. "})
    lang_adapter_layers_decoder: Optional[List[int]] = field(default=None, metadata={"help": "Defines the layers id"
                                                                                      "in which language adapter is"
                                                                                      "added in the decoder. "})
    task_adapter_layers_encoder: Optional[List[int]] = field(default=None, metadata={"help": "Defines the layers id"
                                                                                      "in which task adapters is"
                                                                                      "added in the encoder."})
    task_adapter_layers_decoder: Optional[List[int]] = field(default=None, metadata={"help": "Defines the layers id"
                                                                                      "in which task adapters is"
                                                                                      "added in the decoder."})
    lang_adapter_in_decoder: Optional[bool] = field(default=True, metadata={"help": "If set to false, do not include"
                                                                                    "lang adapters in the decoder."})
    task_adapter_in_decoder: Optional[bool] = field(default=True, metadata={"help": "If set to false, do not include"
                                                                                    "task adapters in the decoder."})
    intrinsic_adapters: Optional[bool] = field(default=False, metadata={"help": "If set to true, computes the "
        "parameters of adapter layers based on an intrinsic vector."})
    keep_adapters_initial_values: Optional[bool] = field(default=True, metadata={"help": "If set to false, we do not"
                                                                                         "use the adapters initial"
                                                                                         "values and they are set to"
                                                                                         "zeros. This option saves on"
                                                                                         "memory."})
    hypercomplex_adapters: Optional[bool] = field(default=False, metadata={"help": "If set, uses the hypercomplex layers"
                                                                                "for adapters."})
    hypercomplex_division: Optional[int] = field(default=8, metadata={"help": "Defines the number to divide the dimensions"
                                                                              "of the linear layer by it."})
    intrinsic_model: Optional[bool] = field(default=False, metadata={"help": "If set, computes all parameters of the "
                                                                             "model with an intrinsic vector."})
    intrinsic_said: Optional[bool] = field(default=False, metadata={"help": "If set, computes the SAID version of the"
                                                                            "model with intrinsic vector."})
    intrinsic_dim: Optional[int] = field(default=100, metadata={"help": "Defines the intrinsic dimensionality."})
    normalize_intrinsic_projections: Optional[bool] = field(default=False, metadata={"help": "If set, normalizes "
        "the intrinsic projection matrices."})
    intrinsic_projection: Optional[str] = field(default="fastfood", metadata={"help": "Defines the type of projection"
        "for intrinsic adapters, it can be random or fastfood."})
    learn_phm: Optional[bool] = field(default=True, metadata={"help": "If set, learns the phm rules in Hypercomplex adapters."})
    normalize_phm_weight: Optional[bool] = field(default=False, metadata={"help": "Weather to normalize the weights of"
                                                                                  "the PHM layer."})
    intrinsic_layer_norms: Optional[bool] = field(default=False, metadata={"help": "If selected, then in case of unfreezing"
        " layernorms for intrinsic_adapters case, it also adds the layernorms parameters inside the parameters given for"
        " the intrinsic projection, and if this is not set, those parameters are not projected with intrinsic vector."})
    hypercomplex_nonlinearity: Optional[str] = field(default="glorot-uniform", metadata={"help": "Defines the nonlinearity for the"
        " hypercomplex adapter layers."})
    shared_phm_rule: Optional[bool] = field(default=False, metadata={"help": "If set, uses a shared phm rules for all"
        " hypercomplex adapter layers."})
    factorized_phm: Optional[bool] = field(default=False, metadata={"help": "If set, it factorizes the weights for the W in"
        " hypercomplex adapters."})
    gradient_clip: Optional[bool] = field(default=False, metadata={"help": "If set, it applies the gradient clipping for hypercomplex adapter layers."})
    phm_clamp: Optional[bool] = field(default=True, metadata={"help": "If set, it clamps the values of the phm layer."})
    shared_W_phm: Optional[bool] = field(default=False, metadata={"help": "If set, shares the W in phm adapter layers between all adapters."})
    factorized_phm_rule: Optional[bool] = field(default=False, metadata={"help": "If set, it factorizes the shared weights for the W in"
        " hypercomplex adapters."})
    phm_c_init: Optional[str] = field(default="normal", metadata={"help": "Initialization for the phm rules."})
    phm_rank: Optional[int] = field(default=1, metadata={"help":"sets the rank for the phm decomposition."})
    phm_init_range: Optional[float] = field(default=0.01, metadata={"help": "defines the phm init range."})
    dropout_at_end: Optional[bool] = field(default=False, metadata={"help": "If set, puts the dropout at the end for adapters."})
    add_adapter_in_feed_forward: Optional[bool] = field(default=True, metadata={"help": "If set, adds adapters in the feedforward."})
    add_adapter_in_self_attention: Optional[bool] = field(default=True, metadata={"help": "If set, adds adapters in the selfattention"})
    prefix_tuning: Optional[bool] = field(default=False, metadata={"help": "If set, uses prefix tuning."})
    prefix_dim: Optional[int] = field(default=100, metadata={"help": "Specifies the prefix embedding dimension."})
    init_prefix_from_vocab: Optional[bool] = field(default=False, metadata={"help": "Initialize prefix from the tokens of pretrained t5-base model."})
    kronecker_prod: Optional[bool] = field(default=False, metadata={"help": "If set, compute the kronecker using another version."})
    bitfit: Optional[bool] = field(default=False, metadata={"help": "If set, we train the bitfit model."})
    freeze_bitfit_lm_head: Optional[bool] = field(default=False, metadata={"help": "If set, freezes the classifier in bitfit."})
    freeze_bitfit_lm_head_all: Optional[bool] = field(default=False, metadata={"help": "If set, freezes the classifier in bitfit."})
