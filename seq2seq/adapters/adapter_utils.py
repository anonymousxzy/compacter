"""Implementation of different utility functions for adapter layers."""

import numpy as np
import os
import torch
import torch.nn as nn
from transformers.activations import get_activation


class Activations(nn.Module):
    def __init__(self, activation_type):
        super().__init__()
        self.f = get_activation(activation_type)

    def forward(self, x):
        return self.f(x)


class TaskHyperNet(nn.Module):
    """This module generates the task-embeddings from the initial fed task embeddings."""

    def __init__(self, config, input_dim, output_dim):
        super(TaskHyperNet, self).__init__()
        self.task_embedding_generator = nn.Sequential(
            nn.Linear(input_dim, config.task_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.task_hidden_dim, output_dim))

    def forward(self, task_embedding):
        task_embedding = task_embedding.view(-1)
        return self.task_embedding_generator(task_embedding).view(-1)


class LayerNormHyperNet(nn.Module):
    """This module generates the weight and bias for the task conditioned layer norm."""

    def __init__(self, config, projection_dim):
        super(LayerNormHyperNet, self).__init__()
        self.weight_generator = nn.Linear(projection_dim, config.input_dim)
        self.bias_generator = nn.Linear(projection_dim, config.input_dim)

    def forward(self, input):
        return self.weight_generator(input), self.bias_generator(input)


class TaskEmbeddingController(nn.Module):
    """Main module controlling task embeddings."""

    def __init__(self, 
             config,
             tasks,
             task_embedding_dim,
             task_embedding_dir,
             task_to_embeddings,
             parametric_task_embedding):
        super(TaskEmbeddingController, self).__init__()
        self.device = config.device
        self.task_embedding_dim = task_embedding_dim 
        self.task_embedding_dir = task_embedding_dir
        self.tasks = tasks
        self.task_to_task_embeddings = {task: task for task in self.tasks}
        self.task_to_embeddings = task_to_embeddings
        if self.task_to_embeddings is not None:
          self.task_to_task_embeddings = self.task_to_embeddings
          self.tasks = self.task_to_task_embeddings.values()
        self.set_task_embeddings(self.tasks, parametric_task_embedding)

    def get_task(self, task):
        return self.task_to_task_embeddings[task]

    def load_or_init_task_embedding(self, task):
        """Loads task embeddings if task_embedding_dir is given or
        initializes them to random."""
        if self.task_embedding_dir is not None:
            task_embedding_path = os.path.join(self.task_embedding_dir, task + ".npy")
            return torch.Tensor(np.load(task_embedding_path)).to(self.device)
        else:
            return torch.Tensor(torch.randn(self.task_embedding_dim)).to(self.device)

    def set_task_embeddings(self, tasks, parametric=False):
        self.task_to_embeddings = {} if not parametric else nn.ParameterDict(dict())
        for task in tasks:
            task_embedding = self.load_or_init_task_embedding(task)
            self.task_to_embeddings[task] = task_embedding if not parametric else nn.Parameter(task_embedding)

    def forward(self, task):
        task_mapped = self.get_task(task)
        task_embedding = self.task_to_embeddings[task_mapped]
        return task_embedding
