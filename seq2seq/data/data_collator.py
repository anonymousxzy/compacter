import numpy as np 
import torch
from typing import List, Union, Dict
from dataclasses import dataclass
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
import sklearn

class Seeder():
    def __init__(self):
        self.seed = 0
    def get(self, num_seeds):
        seeds = np.arange(self.seed, self.seed+num_seeds)
        self.seed = self.seed + num_seeds
        return seeds


def shuffle(data, seeds):
    for i, random_state in enumerate(seeds):
        data[i, :] = sklearn.utils.shuffle(data[i,:], random_state=random_state)
    return data


def random_spans_noise_mask(length, batch, noise_density, mean_noise_span_length, seeds):
    def _random_segmentation(num_items, num_segments, seeds):
       data = (np.arange(num_items-1) < num_segments - 1).astype(int)
       data = np.tile(data, (batch, 1))
       shuffled_data = shuffle(data, seeds)
       shuffled_data = torch.tensor(shuffled_data)
       first_in_segment = torch.cat([torch.zeros(batch, 1), shuffled_data], axis=1)
       segment_id = torch.cumsum(first_in_segment, dim=1)
       _, segment_length = torch.unique_consecutive(segment_id, return_counts=True)
       segment_length = segment_length.reshape(batch, -1)
       return segment_length

    orig_length = length
    # increase length to avoid degeneracy
    length = max(length, 2)
    num_noise_tokens = int(length*noise_density)
    # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_noise_spans = int(num_noise_tokens/mean_noise_span_length)
    # avoid degeneracy by ensuring positive number of noise spans
    num_noise_spans = max(num_noise_spans, 1)
    num_nonnoise_tokens = length - num_noise_tokens
    noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans, seeds[:batch])
    nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans, seeds[batch:])
    stacked = torch.stack([nonnoise_span_lengths, noise_span_lengths], dim=2)
    interleaved_span_lengths = torch.flatten(stacked, start_dim=1, end_dim=2)
    span_starts = torch.cumsum(interleaved_span_lengths, dim=1)[:, :-1]
    span_start_indicator = torch.zeros(batch, length)
    span_start_indicator[torch.arange(batch).unsqueeze(1), span_starts] = 1
    span_num = torch.cumsum(span_start_indicator, dim=1)
    is_noise = torch.eq(span_num % 2, 1.0)
    return is_noise[:, :orig_length]


def noise_span_to_unique_sentinel(tokens,  noise_mask, vocab_size, return_label_mask=False):
    """Sets the tokens selected by the mask to the sentinel tokens."""
    batch = tokens.shape[0]
    prev_token_is_noise = torch.cat([torch.zeros(batch, 1), noise_mask[:, :-1]], axis=1)
    first_noise_tokens = torch.logical_and(noise_mask, torch.logical_not(prev_token_is_noise))
    subsequent_noise_tokens = torch.logical_and(noise_mask, prev_token_is_noise)
    sentinel = vocab_size - torch.cumsum(first_noise_tokens, dim=1)
    # It replaces all the tokens set to true in the first_noise_tokens mask from the sentinel vector.
    # It basically, replaces only the first tokens from each selected span in mask with the sentinel
    # tokens.
    tokens = torch.where(first_noise_tokens, sentinel, tokens)
    # Here in the mask, after not operation, all the subsequent tokens are set to False, meaning that
    # only the tokens which are not in the subsequent mask tokens will be selected.
    result = torch.masked_select(tokens, torch.logical_not(subsequent_noise_tokens)).reshape(batch, -1)

    # During prediction, we need to set the labels for all the masked values in the output sequence to -100.
    # To not compute the loss for them. We therefore need this mask for the target sequence.
    if return_label_mask:
        label_mask = torch.masked_select(first_noise_tokens, torch.logical_not(subsequent_noise_tokens)).reshape(batch, -1)
        return result, label_mask
    return result


def nonnoise_span_to_unique_sentinel(tokens,  noise_mask, vocab_size, return_label_mask=False):
    """Sets the tokens not selected by the mask to the sentinel tokens."""
    return noise_span_to_unique_sentinel(
        tokens=tokens,
        noise_mask=torch.logical_not(noise_mask),
        vocab_size=vocab_size,
        return_label_mask=return_label_mask)

def denoise(tokens, vocab_size, noise_density, mean_noise_span_length, seeds):
    batch, length = tokens.shape
    noise_mask = random_spans_noise_mask(length, batch, noise_density, mean_noise_span_length, seeds)
    inputs = noise_span_to_unique_sentinel(tokens, noise_mask, vocab_size)
    targets, targets_mask = nonnoise_span_to_unique_sentinel(tokens, noise_mask, vocab_size, return_label_mask=True)
    return inputs, targets, targets_mask


class T5PretrainingDataCollator:
    """Implements task-collator to collate the samples in each batch."""
    def __init__(self, tokenizer, model, noise_density, mean_noise_span_length, pad_mask_in_labels):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        assert (
            self.pad_token_id is not None
        ), f"pad_token_id is not defined for ({self.tokenizer.__class__.__name__}), it must be defined."
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.vocab_size = int(len(self.tokenizer))
        self.seeder = Seeder()
        self.model = model
        self.label_pad_token_id = -100
        self.pad_mask_in_labels=pad_mask_in_labels


    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        lang = [d.pop('lang') for d in examples][0]
        task = [d.pop('task') for d in examples][0]
        batch = self.tokenizer.pad(examples, return_tensors="pt")
        batch_size = batch["input_ids"].shape[0]
        seeds = self.seeder.get(batch_size*2)
        # Creates the inputs and targets for T5 pretraining.
        inputs, targets, targets_mask = denoise(batch["input_ids"], self.vocab_size, self.noise_density,
                                  self.mean_noise_span_length, seeds)
        batch["input_ids"] = inputs
        batch["labels"] = targets
        # TODO: I think there is no padding, so all should be 1.
        batch["attention_mask"] = torch.ones_like(inputs)
        if hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            batch["decoder_input_ids"] = self.model.prepare_decoder_input_ids_from_labels(labels=targets)
        # TODO: should this step be done before computing the decoder_input_ids.
        # We set all the masked tokens in the output to -100 to not compute the loss over these tokens.
        if self.pad_mask_in_labels:
            batch["labels"][targets_mask] = self.label_pad_token_id
        batch["lang"] = lang
        batch["task"] = task
        return batch

    # loss should only get computed on non-masked tokens in the labels the masked ones should be -100.


@dataclass
class TaskDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
   def check_uniqueness(self, samples):
        assert len(np.unique(samples)) == 1 

   def __call__(self, features):
        langs = [d.pop('lang') for d in features]
        tasks = [d.pop('task') for d in features]
        # Checks all the tasks and langs should be the same.
        self.check_uniqueness(langs)
        self.check_uniqueness(tasks)
        output = super().__call__(features)
        output["task"] = tasks[0]
        output["lang"] = langs[0]
        return output


@dataclass
class TaskDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def check_uniqueness(self, samples):
        assert len(np.unique(samples)) == 1 
   
    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        langs = [d.pop('lang') for d in examples]
        tasks = [d.pop('task') for d in examples]
        # Checks all the tasks and langs should be the same.
        self.check_uniqueness(langs)
        self.check_uniqueness(tasks)
        batch = super().__call__(examples)
        batch["lang"] = langs[0]
        batch["task"] = tasks[0]
        return batch
