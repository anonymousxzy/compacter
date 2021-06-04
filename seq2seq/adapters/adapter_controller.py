"""Implements Adapter Controller, a module that keeps multiple
layers of Adapters, and controls which adapter layer to use."""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import get_activation

from .adapter_configuration import AdapterConfig, MetaAdapterConfig
from .adapter_modeling import Adapter, AdapterHyperNet, HyperComplexAdapter
from .adapter_utils import LayerNormHyperNet


class AdapterController(nn.Module):
    """Implements Adapter controller module which controls the logics of
    putting adapter layers within transformer's layers."""

    def __init__(self, config):
        super().__init__()
        self.intrinsic_projections_path = os.path.join(config.output_dir, "intrinsic_projections")
        self.config = config
        self.adapters = nn.ModuleDict(dict())
        self.tasks = config.tasks
        self.device = config.device
        self.shared_phm_rule = config.shared_phm_rule
        self.task_to_adapter = {task: task for task in self.tasks}
        # If a dictionary from task to adapter is given, the task is over-written by the given adapters.
        if config.task_to_adapter is not None:
            self.task_to_adapter = config.task_to_adapter
            self.tasks = self.task_to_adapter.values()
        self.hypercomplex_adapters = config.hypercomplex_adapters
        self.adapters = self.construct_adapters(self.tasks)
        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter
        if self.add_layer_norm_before_adapter:
            self.pre_layer_norm = nn.LayerNorm(config.input_dim)
        if self.add_layer_norm_after_adapter:
            self.post_layer_norm = nn.LayerNorm(config.input_dim)
        self.intrinsic_adapters = config.intrinsic_adapters

    def set_task_to_adapter_map(self, mapping):
        self.task_to_adapter = mapping

    def get_task(self, task):
        return task

    def construct_adapters(self, tasks):
        """
        Constructs adapter layers and adds them to a dictionary for the given
        tasks.
        Args:
            tasks: A list of string containing the task names.
        """
        for task in tasks:
            self.adapters[task] = Adapter(self.config) if not self.hypercomplex_adapters\
                else HyperComplexAdapter(self.config)
        return self.adapters

    def disable_adapters(self, tasks):
        """
        Given a list of tasks, it freezes their corresponding adapter layers'
        parameters.
        Args:
           tasks: List of tasks.
        """
        tasks = self.convert_to_list(tasks)
        for task in tasks:
            adapter = self.get_adapter(task)
            for param in adapter.parameters():
                param.requires_grad = False

    def convert_to_list(self, tasks):
        if isinstance(tasks, list):
            return tasks
        return [tasks]

    def enable_adapters(self, tasks):
        """
        Given a list of tasks, it unfreezes their corresponding adapter layers.
        Args:
            tasks: Given list of tasks.
        """
        tasks = self.convert_to_list(tasks)
        for task in tasks:
            adapter = self.get_adapter(task)
            for name, param in adapter.named_parameters():
                if self.config.hypercomplex_adapters and not self.config.learn_phm:
                    if not "phm_rule" in name:
                        param.requires_grad = True
                else:
                    param.requires_grad = True

    def get_adapter(self, task):
        """Given a task returns its corresponding adapter layer.
        Args:
            task: Input task name.
        Returns:
            Adapter layer corresponding to the given task.
        """
        return self.adapters[task]


    def forward(self, inputs, task):
        """Retrieves the adapter layer corresponding to the given
        task. It freezes the adapter layers for all the other tasks
        and call the selected adapter layer.
        Args:
            task: the name of the current task.
            inputs: the inputs to feed in in the adapter layer.
        Returns:
            outputs of the adapter layer."""
        task = self.get_task(task)
        if not self.intrinsic_adapters:
            self.enable_adapters(task)
        # Disable other adapters.
        other_tasks = [x for x in self.tasks if x != task]
        self.disable_adapters(other_tasks)

        adapter = self.get_adapter(task)
        z = self.pre_layer_norm(inputs) if self.add_layer_norm_before_adapter else inputs
        outputs = adapter(z)
        if self.add_layer_norm_after_adapter:
            outputs = self.post_layer_norm(outputs)
        outputs = outputs + inputs
        return outputs


class MetaAdapterController(nn.Module):
    """Implements Meta Adapter controller module, in which
    the adapter layers' weights are generated from a hyper-network.
    In this case, task-embeddings are fixed, they can be initialized
    from a directory (task_embedding_dir) or if not given, the task
    embeddings will be initialized to random."""

    def __init__(self, config):
        super().__init__()
        self.device = config.device
        self.adapters = nn.ModuleDict(dict())
        self.config = config
        self.input_dim = config.input_dim
        self.down_sample_size = self.input_dim // config.reduction_factor
        self.meta_up_sampler = AdapterHyperNet(config, self.input_dim, self.down_sample_size)
        self.meta_down_sampler = AdapterHyperNet(config, self.down_sample_size, self.input_dim)
        self.activation_type = config.non_linearity.lower()
        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter
        self.conditional_layer_norm = config.conditional_layer_norm
        
        projection_dim = config.projected_task_embedding_dim \
            if config.train_task_embeddings else config.task_embedding_dim
        if self.add_layer_norm_after_adapter:
            if self.conditional_layer_norm:
                self.post_layernorm_hypernet = LayerNormHyperNet(config, projection_dim)
            else:
                self.post_layer_norm = nn.LayerNorm(self.input_dim)
        if self.add_layer_norm_before_adapter:
            if self.conditional_layer_norm:
                self.pre_layernorm_hypernet = LayerNormHyperNet(config, projection_dim)
            else:
                self.pre_layer_norm = nn.LayerNorm(self.input_dim)

    def call_adapter(self, inputs, task_embedding):
        weight_up, bias_up = self.meta_up_sampler(task_embedding)
        weight_down, bias_down = self.meta_down_sampler(task_embedding)
        down = F.linear(inputs, weight=weight_down, bias=bias_down)
        middle = get_activation(self.activation_type)(down)
        output = F.linear(middle, weight=weight_up, bias=bias_up)
        return output

    def apply_pre_layer_norm(self, inputs, task_embeddings):
        """Applies pre layer norm to the inputs."""
        if self.conditional_layer_norm:
            weight, bias = self.pre_layernorm_hypernet(task_embeddings)
            return torch.nn.functional.layer_norm(inputs, (self.input_dim,), weight=weight, bias=bias)
        else:
            return self.pre_layer_norm(inputs)

    def apply_post_layer_norm(self, inputs, task_embeddings):
        """Applies post layer norm to the inputs."""
        if self.conditional_layer_norm:
            weight, bias = self.post_layernorm_hypernet(task_embeddings)
            return torch.nn.functional.layer_norm(inputs, (self.input_dim,), weight=weight, bias=bias)
        else:
            return self.post_layer_norm(inputs)

    def forward(self, inputs,  task_embedding):
        """Retrieves the adapter layer corresponding to the given
        task. It freezes the adapter layers for all the other tasks
        and call the selected adapter layer.
        Args:
            task: the name of the current task.
            inputs: the inputs to feed in in the adapter layer.
        Returns:
            outputs of the adapter layer."""
        z = self.apply_pre_layer_norm(inputs, task_embedding) if self.add_layer_norm_before_adapter else inputs
        outputs = self.call_adapter(z, task_embedding)
        if self.add_layer_norm_after_adapter:
            outputs = self.apply_post_layer_norm(outputs, task_embedding)
        outputs = outputs + inputs
        return outputs



class MetaLayersAdapterController(nn.Module):
    """Implements Meta Adapter controller module, in which
    the adapter layers' weights are generated from a unique hyper-network."""

    def __init__(self, config):
        super().__init__()
        self.activation_type = config.non_linearity.lower()
        self.input_dim = config.input_dim
        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter

    def apply_layer_norm(self, inputs, layer_norm_weights, do_float_32):
        """Applies layer norm to the inputs."""
        output = torch.nn.functional.layer_norm(inputs, (self.input_dim,),
                                              weight=self.cast(layer_norm_weights.weight, do_float_32),
                                              bias=self.cast(layer_norm_weights.bias, do_float_32))
        return output

    def cast(self, weight, do_float_32):
        return weight.to(torch.float32) if do_float_32 else weight

    def call_adapter(self, inputs, adapter_weights, do_float_32):
        """Computes the output of the adapter layers."""
        down = F.linear(inputs, weight=self.cast(adapter_weights.down.weight, do_float_32),
                        bias=self.cast(adapter_weights.down.bias, do_float_32))
        middle = get_activation(self.activation_type)(down)
        output = F.linear(middle, weight=self.cast(adapter_weights.up.weight, do_float_32),
                          bias=self.cast(adapter_weights.up.bias, do_float_32))
        return output

    def forward(self, inputs, adapter_weights, do_float_32=False):
        z = self.apply_layer_norm(inputs, adapter_weights.pre_norm, do_float_32) if self.add_layer_norm_before_adapter else inputs
        outputs = self.call_adapter(z, adapter_weights, do_float_32)
        if self.add_layer_norm_after_adapter:
            outputs = self.apply_layer_norm(outputs, adapter_weights.post_norm, do_float_32)
        outputs = outputs + inputs
        return outputs


class AutoAdapterController(nn.Module):
    """Generic adapter controller class to instantiate different adapter
    controller classes."""

    @classmethod
    def get(cls, config):
        if isinstance(config, MetaAdapterConfig):
            return MetaLayersAdapterController(config)
        elif isinstance(config, AdapterConfig):
            return AdapterController(config)
        raise ValueError("Unrecognized adapter config", config)

