"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
import functools
import logging
import torch 
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU' 
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
import sys
import json
import subprocess 
from pathlib import Path
import numpy as np
from typing import Optional, List

from datasets import load_dataset, load_metric, concatenate_datasets
import transformers
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    MBartTokenizer,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import is_main_process, get_last_checkpoint
from seq2seq.utils import get_adapter_config, remove_rank_info_from_argv 
from seq2seq.data import AutoTask
from seq2seq.data import TaskDataCollatorForSeq2Seq
from seq2seq.third_party.trainers import Seq2SeqTrainer
from training_args import AdapterTrainingArguments#, ModelArguments, DataTrainingArguments, TrainingArguments
from seq2seq.utils import modify_model_after_init, save_training_config #get_config, get_model
from dataclasses import dataclass, field
from transformers import Seq2SeqTrainingArguments 
from seq2seq.third_party.models import T5Config, T5ForConditionalGeneration

logger = logging.getLogger(__name__)

def run_command(command):
    output = subprocess.getoutput(command)
    return output


TASK_TO_METRICS = {"mrpc": ["accuracy", "f1"],
                  "cola": ['matthews_correlation'],
                  "stsb": ['pearson', 'spearmanr'],
                  'sst2': ['accuracy'],
                  "mnli": ["accuracy"],
                  "mnli_mismatched": ["accuracy"],
                  "mnli_matched": ["accuracy"],
                  "qnli": ["accuracy"],
                  "rte": ["accuracy"],
                  "wnli": ["accuracy"],
                  "qqp": ["accuracy", "f1"]}


# run_seq2seq parameters.
@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    # TODO:  this is not used now, and I made it automatic, we may want to have both later on.
    #evaluation_metric: Optional[List[str]] = field(default=None,
    #    metadata={"help": "Defines the evaluation metric to compute the average metric for checkpointing."})
    #temperature: Optional[int] = field(default=1, metadata={"help": "Defines the temperature"
    #                                                            "value for sampling across the multiple datasets."})
    print_num_parameters: Optional[bool] = field(default=False, metadata={"help": "If set, print the parameters of "
                                                                                 "the model."})
    do_test: Optional[bool] = field(default=False, metadata={"help": "If set, evaluates the test performance."})
    split_validation_test: Optional[bool] = field(default=False,
                                                  metadata={"help": "If set, for the datasets which do not"
                                                                    "have the test set, we use validation set as their"
                                                                    "test set and make a validation set from either"
                                                                    "splitting the validation set into half (for smaller"
                                                                    "than 10K samples datasets), or by using 1K examples"
                                                                    "from training set as validation set (for larger"
                                                                    " datasets)."})
    compute_time: Optional[bool] = field(default=False, metadata={"help": "If set measures the time."})
    compute_memory: Optional[bool] = field(default=False, metadata={"help": "if set, measures the memory"})
    prefix_length: Optional[int] = field(default=100, metadata={"help": "Defines the length for prefix tuning."})

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    lang: Optional[List[str]] = field(default=None, metadata={"help": "Specify the languages for language adapters."})
    '''
    dataset_name: Optional[List[str]] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    '''
    # TODO: for now lets have one task.
    task_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[List[str]] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    eval_dataset_name: Optional[List[str]] = field(
        default=None, metadata={"help": "The name of the evaluation dataset to use (via the datasets library)."}
    )
    eval_dataset_config_name: Optional[List[str]] = field(
        default=None, metadata={"help": "The configuration name of the evaluation dataset to use (via the datasets library)."}
    )
    test_dataset_name: Optional[List[str]] = field(
        default=None, metadata={"help": "The name of the test dataset to use (via the datasets library)."}
    )
    test_dataset_config_name: Optional[List[str]] = field(
        default=None, metadata={"help": "The configuration name of the test dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    test_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for test target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                    "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                    "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."}
    )
    num_beams: Optional[int] = field(default=None, metadata={"help": "Number of beams to use for evaluation."})
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    task_adapters: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Defines a dictionary from task adapters to the tasks."}
    )
    lang_adapters: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Defines a dictionary from language adapters to the tasks."}
    )
    task_embeddings: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Defines a dictionary from tasks to the tasks embeddings."}
    )
    lang_embeddings: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Defines a dictionary from languages to the language embeddings."}
    )
    data_seed: Optional[int] = field(default=42, metadata={"help": "seed used to shuffle the data."})

    def __post_init__(self):
        # TODO: modified for now.
        #if self.dataset_name is None:
        if self.task_name is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length
        if self.test_max_target_length is None:
            self.test_max_target_length = self.max_target_length

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    # print("##### hostname ", run_command("hostname"))
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments,
                               AdapterTrainingArguments))
    if len(sys.argv) > 2 and sys.argv[1].startswith("--local_rank") and (sys.argv[2].endswith(".json")):
        rank_info = remove_rank_info_from_argv(sys.argv)
        args_dict = json.loads(Path(sys.argv[1]).read_text())
        args_dict.update(rank_info)
        model_args, data_args, training_args, adapter_args = parser.parse_dict(args_dict)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()


    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        print("#### last_checkpoint ", last_checkpoint)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            '''
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
            '''
            pass 
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files in the summarization task, this script will use the first column for the full texts and the
    # second column for the summaries (unless you specify column names for this with the `text_column` and
    # `summary_column` arguments).
    # For translation, only JSON files are supported, with one field named "translation" containing two keys for the
    # source and target languages (unless you adapt what follows).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    #TODO: check here.
    config = T5Config.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    print(config)
    #config = get_config(model_args)
    #############################################################
    config.train_task_adapters = adapter_args.train_task_adapters
    config.train_lang_adapters = adapter_args.train_lang_adapters
    config.prefix_tuning = adapter_args.prefix_tuning
    adapter_config = get_adapter_config(adapter_args, data_args, training_args, config)
    #############################################################
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        adapter_config=adapter_config
    )
    model.resize_token_embeddings(len(tokenizer))
    #model = get_model(model_args, config, adapter_config)
    model = modify_model_after_init(model, training_args, adapter_args)

    #TODO: check here.
    data_args.dataset_name = [data_args.task_name]
    assert len(data_args.dataset_name) == len(data_args.dataset_config_name)
    if data_args.eval_dataset_name is not None:
        assert len(data_args.eval_dataset_name) == len(data_args.eval_dataset_config_name)
    if data_args.test_dataset_name is not None:
        assert len(data_args.test_dataset_name) == len(data_args.test_dataset_config_name)

    # Temporarily set max_target_length for training.
    #max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False
    
    def preprocess_function(examples, max_target_length):
        model_inputs = tokenizer(examples['source'], max_length=data_args.max_source_length,
                                 padding=padding, truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['target'], max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    column_names = ['source', 'target']
    performance_metrics = {}
    if training_args.do_train:
        print("#### train dataset #######")
        train_datasets = [AutoTask.get(dataset_name,
                                       dataset_config_name,
                                       seed=data_args.data_seed).get(
            split="train", 
            split_validation_test=training_args.split_validation_test,
            add_prefix=False if (adapter_args.train_task_adapters or adapter_args.train_lang_adapters) else True,
            n_obs=data_args.max_train_samples)
            for dataset_name, dataset_config_name\
            in zip(data_args.dataset_name, data_args.dataset_config_name)]
        max_target_lengths = [AutoTask.get(dataset_name, dataset_config_name).get_max_target_length(\
            tokenizer=tokenizer, default_max_length=data_args.max_target_length)\
            for dataset_name, dataset_config_name in zip(data_args.dataset_name, data_args.dataset_config_name)]
        for i, train_dataset in enumerate(train_datasets):
            train_datasets[i] = train_datasets[i].map(
                functools.partial(preprocess_function, max_target_length=max_target_lengths[i]),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
            )
        train_dataset_sizes = [len(train_dataset) for train_dataset in train_datasets]
        train_dataset = concatenate_datasets(train_datasets)

    if training_args.do_eval:
        # TODO: for different langauges need a change of key to eval_dataset+"_"+eval_dataset_config
        eval_datasets = {eval_dataset: AutoTask.get(eval_dataset,
                                                                            eval_dataset_config,
                                                                            seed=data_args.data_seed).get(
            split="validation", 
            split_validation_test=training_args.split_validation_test,
            add_prefix=False if (adapter_args.train_task_adapters or adapter_args.train_lang_adapters) else True,
            n_obs=data_args.max_val_samples)
            for eval_dataset, eval_dataset_config in zip(data_args.eval_dataset_name, data_args.eval_dataset_config_name)}
        # TODO: till now max target length is computed per training examples.
        max_target_lengths = [AutoTask.get(dataset_name, dataset_config_name).get_max_target_length( \
            tokenizer=tokenizer, default_max_length=data_args.max_target_length) \
            for dataset_name, dataset_config_name in zip(data_args.eval_dataset_name, data_args.eval_dataset_config_name)]
        for k, name in enumerate(eval_datasets):
            eval_datasets[name] = eval_datasets[name].map(
                    functools.partial(preprocess_function, max_target_length=max_target_lengths[k]),
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
            )
    if training_args.do_test:
        # TODO: for different langauges need a change of key to eval_dataset+"_"+test_dataset_config
        test_datasets = {test_dataset: AutoTask.get(test_dataset,
                                                                            test_dataset_config,
                                                                            seed=data_args.data_seed).get(
            split="test", 
            split_validation_test=training_args.split_validation_test,
            add_prefix=False if (adapter_args.train_task_adapters or adapter_args.train_lang_adapters) else True,
            n_obs=data_args.max_test_samples)
            for test_dataset, test_dataset_config in zip(data_args.test_dataset_name, data_args.test_dataset_config_name)}
        max_target_lengths = [AutoTask.get(dataset_name, dataset_config_name).get_max_target_length( \
            tokenizer=tokenizer, default_max_length=data_args.max_target_length) \
            for dataset_name, dataset_config_name in zip(data_args.test_dataset_name, data_args.test_dataset_config_name)]
        for k, name in enumerate(test_datasets):
            test_datasets[name] = test_datasets[name].map(
                    functools.partial(preprocess_function, max_target_length=max_target_lengths[k]),
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
            )

    # TODO: can we do the data_collator part more efficiently.
    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = TaskDataCollatorForSeq2Seq(
            tokenizer,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    # Metric
    # TODO: for now lets assume we have one metric only.
    eval_metrics = [AutoTask.get(dataset_name, dataset_config_name).metric\
        for dataset_name, dataset_config_name in zip(data_args.dataset_name, data_args.dataset_config_name)][0]
    eval_metrics_names = [AutoTask.get(dataset_name, dataset_config_name).metric_names \
                    for dataset_name, dataset_config_name in
                    zip(data_args.dataset_name, data_args.dataset_config_name)][0]
    #eval_metrics = metric_names[0] # _name
    print("#### metric ", eval_metrics)
    '''
    if metric_name in ["sacrebleu", "rouge"]:
        metric = load_metric(metric_name)
    '''
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Some simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        '''
        if metric_name == "sacrebleu":
            decoded_labels = [[label] for label in decoded_labels]

        if metric in ["sacrebleu", "rouge"]:
            metric = load_metric(metric_name)
            result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        else:
        '''
        # TODO: eval metrics needs to be two
        result = {}
        for metric in eval_metrics:
            result.update(metric(decoded_preds, decoded_labels))
        '''
        if metric_name == "accuracy":
            result = accuracy(decoded_preds, decoded_labels)
        elif metric_name == "squad":
            result = metric.squad(decoded_preds, decoded_labels)
        else:
            result = metric.compute(predictions=decoded_preds, references=decoded_labels)
      
        # Extract a few results from ROUGE
        if metric_name == "rouge":
            result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        elif metric_name == "sacrebleu":
            result = {"bleu": result["score"]}
        '''
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    # TODO: evaluate needs to get changed to work with eval_strategy also
    #assert training_args.evaluation_metric is not None, "evaluation metric cannot be None."
    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=list(eval_datasets.values())[0] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        # TODO: for now only works for one dataset.
        evaluation_metrics = TASK_TO_METRICS[data_args.dataset_name[0]] #task_name]
        #adapter_config=adapter_config,
        #train_dataset_sizes=train_dataset_sizes if training_args.do_train else None,
        #evaluation_metric=eval_metrics_names, #training_args.evaluation_metric
        #adapter_args=adapter_args
    )
    # Saves training config. 
    if trainer.is_world_process_zero():
       os.makedirs(training_args.output_dir, exist_ok=True)
       save_training_config(sys.argv[1], training_args.output_dir)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        if training_args.compute_time:
            torch.cuda.synchronize()  # wait for move to complete
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
        if training_args.compute_time:
            end.record()
            torch.cuda.synchronize()  # wait for all_reduce to complete
            total_time = start.elapsed_time(end)/(1000*60)
            performance_metrics.update({"total_time in minutes ": total_time})
        
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if torch.cuda.is_available() and training_args.compute_memory:
        peak_memory = (torch.cuda.max_memory_allocated() / 1024 ** 2)/1000
        print(
            "Memory utilization",
            peak_memory,
            "GB"
        )
        performance_metrics.update({"peak_memory": peak_memory})
    if training_args.compute_memory or training_args.compute_time:
        print(performance_metrics)
        trainer.save_metrics("performance", performance_metrics)
    
    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        for task, eval_dataset in eval_datasets.items(): #zip(eval_datasets, data_args.eval_dataset_name):
            metrics = trainer.evaluate(eval_dataset=eval_dataset,
               max_length=data_args.val_max_target_length, num_beams=data_args.num_beams, # metric_key_prefix="eval"
            )
            #max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
            #metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    # Test
    if training_args.do_test:
        logger.info("*** Test ***")
        for task, test_dataset in test_datasets.items(): #zip(test_datasets, data_args.test_dataset_name):
            metrics = trainer.evaluate(eval_dataset=test_dataset,
              max_length=data_args.test_max_target_length, num_beams=data_args.num_beams, #metric_key_prefix="test"
            )
            #max_test_samples = data_args.max_test_samples if data_args.max_test_samples is not None else len(test_dataset)
            #metrics["test_samples"] = min(max_test_samples, len(test_dataset))
            trainer.log_metrics("test", metrics)
            trainer.save_metrics("test", metrics)



    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
