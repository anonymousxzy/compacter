"""Implements an Adapter and Hyper-adapter Layers."""
import torch.nn as nn
import torch
from .adapter_utils import Activations, LayerNormHyperNet, TaskHyperNet
from .adapter_outputs import (SamplerOutput, LayerNormOutput,
                              AdapterBlockOutput, AdapterOutput)
from seq2seq.hypercomplex.layers import PHMLinear
import numpy as np

class Adapter(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.down_sample_size = self.input_dim // config.reduction_factor
        self.activation = Activations(config.non_linearity.lower())
        self.down_sampler = nn.Linear(self.input_dim, self.down_sample_size) 
        self.up_sampler = nn.Linear(self.down_sample_size, self.input_dim) 

    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        output = self.up_sampler(z)
        return output 


# TODO: update the documentation and also choose a better names for the variables.
class HyperComplexAdapter(nn.Module):
    """Hypercomplex Adapter layer, in which the weights of up and down sampler modules
    are parameters are 1/n times of the conventional adapter layers, where n is
    hypercomplex division number."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.down_sample_size = self.input_dim // config.reduction_factor
        self.activation = Activations(config.non_linearity.lower())
        self.down_sampler = PHMLinear(in_features=self.input_dim,
                                      out_features=self.down_sample_size,
                                      bias=True, clamp=config.phm_clamp, c_init=config.phm_c_init,
                                      phm_dim=config.hypercomplex_division, learn_phm=config.learn_phm,
                                      w_init=config.hypercomplex_nonlinearity,
                                      normalize=config.normalize_phm_weight, shared_phm_rule=config.shared_phm_rule,
                                      factorized_phm=config.factorized_phm, shared_W_phm=config.shared_W_phm,
                                      factorized_phm_rule=config.factorized_phm_rule,
                                      phm_rank=config.phm_rank, phm_init_range=config.phm_init_range,
                                      kronecker_prod=config.kronecker_prod)
        self.up_sampler = PHMLinear(in_features=self.down_sample_size,
                                    out_features=self.input_dim, 
                                    bias=True, clamp=config.phm_clamp, c_init=config.phm_c_init, 
                                    phm_dim=config.hypercomplex_division, learn_phm=config.learn_phm,
                                    w_init=config.hypercomplex_nonlinearity,
                                    normalize=config.normalize_phm_weight, shared_phm_rule=config.shared_phm_rule,
                                    factorized_phm=config.factorized_phm, shared_W_phm=config.shared_W_phm,
                                    factorized_phm_rule=config.factorized_phm_rule,
                                    phm_rank=config.phm_rank, phm_init_range=config.phm_init_range,
                                    kronecker_prod=config.kronecker_prod)
        self.gradient_clip = config.gradient_clip

    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        output = self.up_sampler(z)
        
        if self.gradient_clip:
            # Clip the gradients.
            phm_rule_list = []
            for name, module in self.named_modules():
              for phm_rule_name in ["phm_rule", "phm_rule_left", "phm_rule_right"]:  
                if hasattr(module, phm_rule_name):
                    m = getattr(module, phm_rule_name)
                    phm_rule_list.append(m)
            torch.nn.utils.clip_grad_norm_(phm_rule_list, max_norm=1.0, norm_type=2.0)  # does operation in-place.
 
        return output


class AdapterHyperNet(nn.Module):
    """This module generates the weights for the meta adapter layers."""

    def __init__(self, config, input_dim, output_dim):
        super(AdapterHyperNet, self).__init__()
        self.hidden_dim = config.hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.train_task_embeddings = config.train_task_embeddings
        self.task_embedding_dim = config.projected_task_embedding_dim if \
            config.train_task_embeddings else config.task_embedding_dim
        self.one_layer_adapter_hyper_net = config.one_layer_adapter_hyper_net
        self.adapter_hyper_net_with_bias = config.adapter_hyper_net_with_bias
        self.one_layer_adapter_hyper_net_with_linear = config.one_layer_adapter_hyper_net_with_linear
        # Considers weight and bias parameters for generating adapter weights.
        if self.one_layer_adapter_hyper_net:
            self.weight_generator = nn.Parameter(torch.Tensor(self.task_embedding_dim,
                                                              self.input_dim, self.output_dim))
            if self.adapter_hyper_net_with_bias:
                self.bias_generator = nn.Parameter(torch.Tensor(self.task_embedding_dim, self.input_dim))
            else:
                self.register_parameter('bias_generator', None)
            nn.init.normal_(self.weight_generator, std=1e-2)
            if self.bias_generator is not None:
                nn.init.zeros_(self.bias_generator)
        # Generates the adapter's weight with linear layers.
        elif self.one_layer_adapter_hyper_net_with_linear:
            self.weight_generator = nn.Sequential(
                nn.Linear(self.task_embedding_dim, self.input_dim * self.output_dim))
            self.bias_generator = nn.Sequential(
                nn.Linear(self.task_embedding_dim, self.input_dim))
        else:
            # Generates the adapter's weight with two linear layers.
            self.weight_generator = nn.Sequential(
                nn.Linear(self.task_embedding_dim, self.hidden_dim),
                nn.Linear(self.hidden_dim, self.input_dim * self.output_dim))
            self.bias_generator = nn.Sequential(
                nn.Linear(self.task_embedding_dim, self.hidden_dim),
                nn.Linear(self.hidden_dim, self.input_dim))

    def forward(self, task_embedding):
        task_embedding = task_embedding.view(-1)
        if self.one_layer_adapter_hyper_net:
            bias = None
            weight = torch.matmul(task_embedding, self.weight_generator.view(self.task_embedding_dim, -1)
                                  ).view(self.input_dim, self.output_dim)
            if self.adapter_hyper_net_with_bias:
                bias = torch.matmul(task_embedding, self.bias_generator)
        else:
            weight = self.weight_generator(task_embedding).view(self.input_dim, self.output_dim)
            bias = self.bias_generator(task_embedding).view(-1)
        return weight, bias


class AdapterLayersHyperNet(nn.Module):
    """This module generates the weights for all the meta adapter layers
    given the task embeddings and layer id."""

    def __init__(self, input_dim, output_dim, projection_dim):
        super(AdapterLayersHyperNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_generator = nn.Linear(projection_dim, self.input_dim*self.output_dim)
        self.bias_generator = nn.Linear(projection_dim, self.input_dim)

    def forward(self, embeddings):
        weight = self.weight_generator(embeddings).view(self.input_dim, self.output_dim)
        bias = self.bias_generator(embeddings).view(-1)
        return SamplerOutput(weight=weight, bias=bias)


class ConditionedEmbedding(nn.Module):
    """Generates an embedding based on the task embedding, id of the layer, and the position of
    Adapter layers which is shown with block_type."""
    def __init__(self, config, task_embedding_dim, num_layers):
        super(ConditionedEmbedding, self).__init__()
        self.layer_norm_epsilon = 1e-6
        self.device = config.device
        self.layer_id_embeddings = nn.Embedding(num_layers, task_embedding_dim).to(self.device)
        # This is 2 types of adapters for feed-forward, and self-attention.
        self.adapters_block_type = nn.Embedding(2, task_embedding_dim).to(self.device)
        self.task_hypernet = TaskHyperNet(config, task_embedding_dim*3, task_embedding_dim)
        self.unique_hyper_net_layer_norm = config.unique_hyper_net_layer_norm
        if self.unique_hyper_net_layer_norm:
            self.LayerNorm = nn.LayerNorm(task_embedding_dim, eps=self.layer_norm_epsilon)

    def forward(self, task_embedding, layer_id, block_type):
        """Concatenates the task embedding with the embedding for the layer id and
        returns the final joint embedding."""
        layer_id_tensor = torch.tensor([layer_id], dtype=torch.long, device=self.device)
        layer_embedding = self.layer_id_embeddings(layer_id_tensor)
        type_id_tensor = torch.tensor([block_type], dtype=torch.long, device=self.device)
        type_embedding = self.adapters_block_type(type_id_tensor)
        layer_embedding = layer_embedding.view(-1)
        type_embedding = type_embedding.view(-1)
        # TODO: this must be wrong for hyperformer work.
        embeddings = torch.cat([task_embedding.view(1, -1), layer_embedding.view(1, -1),
                                type_embedding.view(1, -1)], axis=0)
        embeddings = self.task_hypernet(embeddings.view(-1))
        if self.unique_hyper_net_layer_norm:
            embeddings = self.LayerNorm(embeddings)
        return embeddings


class AdapterLayersOneHyperNetController(nn.Module):
    """This modules contains the hyper-nets for the feed forward
    and self-attention modules and it generates the adapter's weights and
    layer norm's weights for all the layers of transformers."""

    def __init__(self, config, task_embedding_dim, num_layers, reduction_factor):
        super(AdapterLayersOneHyperNetController, self).__init__()
        self.device = config.device
        self.conditional_embedder = ConditionedEmbedding(config, task_embedding_dim, num_layers)
        self.input_dim = config.input_dim
        self.down_sample_size = self.input_dim // reduction_factor
        # Defines the adapters hyper-nets.
        self.up_sampler_hyper_net = AdapterLayersHyperNet(self.input_dim, self.down_sample_size,
            task_embedding_dim)
        self.down_sampler_hyper_net = AdapterLayersHyperNet(self.down_sample_size, self.input_dim,
            task_embedding_dim)
        # Defines the layer norms' hyper-nets.
        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter
        if self.add_layer_norm_before_adapter:
            self.pre_layernorm_hypernet = LayerNormHyperNet(config, task_embedding_dim)
        if self.add_layer_norm_after_adapter:
            self.post_layernorm_hypernet = LayerNormHyperNet(config, task_embedding_dim)


    def project_params_on_intrinsic_vec(self, task):
        """Updates the parameters of the module based on the intrinsic vector."""
        for name, base, localname in self.name_base_localname[task]:
            init_shape = self.initial_shapes[task][name]
            desired_dim = np.prod(init_shape)
            projection = self.intrinsic_projector.compute_projection(self.intrinsic_vec, desired_dim=desired_dim,
                param_list=self.projection_mats[task][name]).view(init_shape)
            setattr(base, localname, projection)

    def add_parameters(self, module, task):
        """Generates random matrices and keeps projection matrices and initial values."""
        self.initial_shapes[task] = {}
        self.projection_mats[task] = {}
        self.name_base_localname[task] = []
        for name, param in module.named_parameters():
            param.requires_grad = False
            self.initial_shapes[task][name] = param.size()
            # Computes the desired dimension.
            desired_dim = np.prod(param.shape)
            self.projection_mats[task][name] = self.intrinsic_projector.get_projection_vars(desired_dim)
            # Computes the localname.
            base, localname = module, name
            while "." in localname:
                prefix, localname = localname.split(".", 1)
                base = base.__getattr__(prefix)
            # adds global name, base model, and localname to the list.
            self.name_base_localname[task].append((name, base, localname))

    def get_output_block(self, embedding):
        down = self.down_sampler_hyper_net(embedding)
        up = self.up_sampler_hyper_net(embedding)
        return AdapterOutput(up=up, down=down)

    def get_norm_outputs(self, layernorm_hypernet, feed_forward_embeddings,
                        self_attention_embeddings):
            weight, bias = layernorm_hypernet(feed_forward_embeddings)
            feed_forward_norm = LayerNormOutput(weight=weight, bias=bias)
            weight, bias = layernorm_hypernet(self_attention_embeddings)
            self_attention_norm = LayerNormOutput(weight=weight, bias=bias)
            return feed_forward_norm, self_attention_norm

    def forward(self, task_embedding, layer_id):
        feed_forward_embeddings = self.conditional_embedder(task_embedding, layer_id, 0)
        self_attention_embeddings = self.conditional_embedder(task_embedding, layer_id, 1)

        # Generates the weights for feed-forward and self-attention.
        feed_forward_output = self.get_output_block(feed_forward_embeddings)
        self_attention_output = self.get_output_block(self_attention_embeddings)

        # Generates the weights and baises for pre and post layer norms.
        if self.add_layer_norm_before_adapter:
            feed_forward_output.pre_norm, self_attention_output.pre_norm =\
                self.get_norm_outputs(self.pre_layernorm_hypernet,
                feed_forward_embeddings, self_attention_embeddings)

        if self.add_layer_norm_after_adapter:
            feed_forward_output.post_norm, self_attention_output.post_norm =\
                self.get_norm_outputs(self.post_layernorm_hypernet,
                feed_forward_embeddings, self_attention_embeddings) 

        return AdapterBlockOutput(feed_forward=feed_forward_output, self_attention=self_attention_output)


