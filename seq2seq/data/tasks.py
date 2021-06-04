from collections import OrderedDict
import abc
import functools
from typing import Callable, List, Mapping
from utils import pad_punctuation
from seq2seq.metrics import metrics
from .utils import get_spans, round_stsb_target
import datasets
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)

lang_to_lang = {
    "en": "english",
    "ro": "romanian",
    "ar": "arabic",
    "bn": "bengali",
    "fi": "finnish",
    "id": "indonesian",
    "ko": "korean",
    "ru": "russian",
    "sw": "swahili",
    "te": "telugu"
}

class AbstractTask(abc.ABC):
    name = NotImplemented
    config = NotImplemented
    prefix = NotImplemented
    preprocessor: Callable = NotImplemented
    metric = NotImplemented
    metric_names = NotImplemented
    split_map = None
    labels_list = None
    split_to_data_split: Mapping[str, str] = \
        {"train": "train", "validation": "validation", "test": "test"}
    small_datasets_without_all_splits = ["cola", "wnli", "rte", "trec", "superglue-cb", "sick",
                                         "mrpc", "stsb", "imdb", "boolq", "commonsense_qa", "superglue-boolq"]
    large_data_without_all_splits = ["yelp_polarity", "qqp", "qnli",
                                     "social_i_qa", "cosmos_qa", "winogrande", "hellaswag", "sst2"]

    def __init__(self, config, seed=42):
        self.config = config
        self.seed = seed

    def get_max_target_length(self, tokenizer, default_max_length):
        if self.labels_list is not None:
            return max([len(tokenizer.encode(label)) for label in self.labels_list])
        return default_max_length

    def seq2seq_format(self, sources: List[str],
                       targets: List[str],
                       add_prefix: bool=False,
                       prefix: str=None,
                       language="en"):
        src_prefix = self.name if prefix is None else prefix
        sources = [src_prefix]+sources if add_prefix else sources
        return {'source': ' '.join(sources),
                'target': ' '.join(targets),
                'task': self.name,
                'lang': language}

    def check_n_obs(self, n_obs, total_size):
        if n_obs is not None and n_obs > total_size:
            n_obs = total_size
            logger.warning("n_obs is set to %s", n_obs)
        return n_obs
   
    def shuffled_indices(self, dataset):
        num_samples = len(dataset)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return torch.randperm(num_samples, generator=generator).tolist()

    def subsample(self, dataset, n_obs=None, indices=None):
        """
        Given a dataset returns the subsampled dataset.
        :param n_obs: the number of samples of the subsampled dataset.
        :param indices: indices to select the samples from, if not given, indices are computed
        from by shuffling the given dataset.
        :return: subsampled dataset.
        """
        num_samples = len(dataset)
        n_obs = self.check_n_obs(n_obs, num_samples)
        if indices is None:
           indices = self.shuffled_indices(dataset)
        indices = indices[:n_obs]
        return dataset.select(indices)

    def load_dataset(self, split: int):
        return datasets.load_dataset(self.name, self.config, split=split, script_version="master")

    def get_split_indices(self, split, dataset, validation_size):
        indices = self.shuffled_indices(dataset)
        if split == "validation":
            return indices[:validation_size]
        else:
            return indices[validation_size:]
            
    def get(self, split, add_prefix=True, n_obs=None, split_validation_test=False):
        # For small datasets (n_samples < 10K) without test set, we divide validation set to
        # half, use one half as test set and one half as validation set.
        if split_validation_test and self.name in self.small_datasets_without_all_splits \
                and split != "train":
            mapped_split = self.split_to_data_split["validation"]
            dataset = self.load_dataset(split=mapped_split)
            indices = self.get_split_indices(split, dataset, validation_size=len(dataset)//2)
            dataset = self.subsample(dataset, n_obs, indices)
        # For larger datasets (n_samples > 10K), we divide training set into 1K as
        # validation and the rest as training set, keeping the original validation
        # set as the test set.
        elif split_validation_test and self.name in self.large_data_without_all_splits \
                and split != "test":
            dataset = self.load_dataset(split="train")
            indices = self.get_split_indices(split, dataset, validation_size=1000)
            dataset = self.subsample(dataset, n_obs, indices)
        else:
            mapped_split = self.split_to_data_split[split]
            dataset = self.load_dataset(split=mapped_split)
            # shuffles the data and samples it.
            if n_obs is not None:
                dataset = self.subsample(dataset, n_obs)
        return dataset.map(functools.partial(self.preprocessor, add_prefix=add_prefix),
                           remove_columns=dataset.column_names)

class PAWSX(AbstractTask):
    name = "paws-x"
    metric = [metrics.accuracy]
    labels_list = ["0", "1"]

    def preprocessor(self, example, add_prefix=True):
        source = ["sentence1:", example["sentence1"],
                  "sentence2:", example["sentence2"]]
        target = [str(example["label"])]
        return self.seq2seq_format(source, target, add_prefix, language=self.config)


class XNLI(AbstractTask):
    name = "xnli"
    metric = [metrics.accuracy]
    labels_list = ["0", "1", "2"]

    def preprocessor(self, example, add_prefix=True):
        source = ["premise:", example["premise"],
                  "hypothesis:", example["hypothesis"]]
        target = [str(example["label"])]
        return self.seq2seq_format(source, target, add_prefix, language=self.config)


class WikiAnn(AbstractTask):
    name="wikiann"
    metric = [metrics.accuracy]
    # We considered only the languages used in mt5 as the valid languages.
    valid_langs = [
    "af", "ar", "bg", "bn", "de", "el", "en", "es", "et", "eu", "fa", "fi",
    "fr", "he", "hi", "hu", "id", "it", "ja", "jv", "ka", "kk", "ko", "ml",
    "mr", "ms", "my", "nl", "pt", "ru", "sw", "ta", "te", "th", "tl", "tr",
    "ur", "vi", "yo", "zh"]    
    ner_tags = {
       0:"O",
       1:"B-PER",
       2:"I-PER",
       3:"B-ORG",
       4:"I-ORG",
       5:"B-LOC",
       6:"I-LOC"
    }

    def preprocessor(self, example, add_prefix=True):
        tokens = example['tokens']
        tags = example['ner_tags']
        tags = [self.ner_tags[i] for i in tags]
        spans = get_spans(tokens, tags)
        source = ["tag:"]+tokens
        target = [' $$ '.join(spans)]
        return self.seq2seq_format(source, target, add_prefix, language=self.config)



class WMT16(AbstractTask):
    name = "wmt16"
    metric = "sacrebleu"

    def preprocessor(self, example, add_prefix=True):
        source_lang, target_lang = self.config.split("-")
        source = [example['translation'][source_lang]]
        target = [example['translation'][target_lang]]
        # TODO: should it be a source lang?
        return self.seq2seq_format(source, target, add_prefix,
                                   prefix="translate {lang_to_lang[source_lang]} to {lang_to_lang[target_lang]}",
                                   language=source_lang)


# TODO: for now I return back the language which is not english as the output.
#   to make the model monolingual.
class OPUS100(AbstractTask):
    name = "opus100"
    metric = "accuracy"

    def preprocessor(self, example, add_prefix=True):
        source_lang, target_lang = self.config.split("-")
        non_english_lang = source_lang if source_lang != "en" else target_lang
        source = [example['translation'][non_english_lang]]
        target = []
        return self.seq2seq_format(source, target, add_prefix,
                                   prefix="", language=non_english_lang)

# TODO: there is a post processor in mt5 model and also what will happen with a list of the answers.
class MLQA(AbstractTask):
    name = "mlqa"
    metric = [metrics.squad]

    def load_dataset(self, split):
        return datasets.load_dataset(self.name, 'mlqa-translate-train.'+self.config,
            split=split, script_version="master")

    def preprocessor(self, example, add_prefix):
        assert len(example['answers']['text'])==1, "the length of the answer should be one."
        answer = pad_punctuation(example['answers']['text'][0])
        question = pad_punctuation(example['question'])
        context = pad_punctuation(example['context'])
        source = ["question:", question,
                  "context:", context]
        target = [answer]
        return self.seq2seq_format(source, target, add_prefix, language=self.config)


class TyDiQA(AbstractTask):
    name = "tydiqa"
    metric = [metrics.squad]

    def load_dataset(self, split):
        dataset = datasets.load_dataset(self.name, 'secondary_task', split=split, script_version="master")
        return dataset.filter(lambda x: x["id"].startswith(lang_to_lang[self.config]))

    def preprocessor(self, example, add_prefix):
        answer = pad_punctuation(example['answers']['text'][0])
        question = pad_punctuation(example['question'])
        context = pad_punctuation(example['context'])
        source = ["question:", question,
                  "context:", context]
        target = [answer]
        return self.seq2seq_format(source, target, add_prefix, language=self.config)


class CC100(AbstractTask):
    name = "cc100"
    metric = "loss"

    def preprocessor(self, example, add_prefix=True):
        source = [example['text']]
        target = []
        return self.seq2seq_format(source, target, add_prefix,
                                   prefix="", language=self.config)

class Squad(AbstractTask):
    name = "squad"
    metric = [metrics.squad]

    def load_dataset(self, split):
        return datasets.load_dataset(self.name, split=split, script_version="master")

    def preprocessor(self, example, add_prefix):
        answer = pad_punctuation(example['answers']['text'][0])
        question = pad_punctuation(example['question'])
        context = pad_punctuation(example['context'])
        source = ["question:", question,
                  "context:", context]
        target = [answer]
        return self.seq2seq_format(source, target, add_prefix, language="en")

class Paws(AbstractTask):
    name = "paws"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset(self.name, 'labeled_final', split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example["sentence1"], "sentence2:", example["sentence2"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)

class SNLI(AbstractTask):
    name = "snli"
    labels_list = ["0", "1", "2"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset(self.name, split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example["premise"], "hypothesis:", example["hypothesis"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class Scitail(AbstractTask):
    name = "scitail"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    label_map = {"entailment": 0, "neutral": 1}

    def map_label(self, label):
       return self.label_map[label]

    def load_dataset(self, split):
        return datasets.load_dataset("scitail", "snli_format", split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'], "sentence2:", example["sentence2"]]
        # To increase the transfer performance, we modified the targets to be similar to other datasets.
        tgt_texts = [str(self.map_label(example['gold_label']))]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class MRPC(AbstractTask):
    name = "mrpc"
    labels_list = ["0", "1"]
    metric = [metrics.f1_score_with_invalid, metrics.accuracy]
    metric_names = ["f1", "accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'mrpc', split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class COLA(AbstractTask):
    name = "cola"
    labels_list = ["0", "1"]
    metric = [metrics.matthews_corrcoef]
    metric_names = ["matthews_correlation"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'cola',
                                     split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", example['sentence']]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SST2(AbstractTask):
    name = "sst2"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'sst2',
                                     split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", example['sentence']]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class STSB(AbstractTask):
    name = "stsb"
    labels_list = [str(np.round(label, decimals=1)) for label in np.arange(0, 5.2, 0.2)]
    metric = [metrics.pearson_corrcoef, metrics.spearman_corrcoef]
    metric_names = ["pearson", "spearmanr"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'stsb',
                                     split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(round_stsb_target(example['label']))]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class QQP(AbstractTask):
    name = "qqp"
    labels_list = ["0", "1"]
    metric = [metrics.f1_score_with_invalid, metrics.accuracy]
    metric_names = ["f1", "accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'qqp',
                                     split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question1:", example['question1'],
                     "question2:", example["question2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class MNLI(AbstractTask):
    name = "mnli"
    labels_list = ["0", "1", "2"]
    split_to_data_split = {"train": "train",
                           "validation": "validation_mismatched",
                           "test": "validation_matched"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]


    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'mnli', split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example['premise'],
                     "hypothesis", example["hypothesis"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class QNLI(AbstractTask):
    name = "qnli"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'qnli', split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question:", example['question'],
                     "sentence:", example["sentence"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)

class RTE(AbstractTask):
    name = "rte"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'rte',
                                     split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class WNLI(AbstractTask):
    name = "wnli"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'wnli', split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


TASK_MAPPING = OrderedDict(
    [
        ('paws-x', PAWSX),
        ('wmt16', WMT16),
        ('opus100', OPUS100),
        ('xnli', XNLI),
        ('wikiann', WikiAnn),
        ('mlqa', MLQA),
        ('cc100', CC100),
        ('tydiqa', TyDiQA),
        ('squad', Squad),
        ('paws', Paws),
        ('snli', SNLI),
        ('scitail', Scitail),
        ('mrpc', MRPC),
        ('cola', COLA),
        ('sst2', SST2),
        ('qnli', QNLI),
        ('rte', RTE),
        ('wnli', WNLI),
        ('mnli', MNLI),
        ('qqp', QQP),
        ('stsb', STSB)
    ]
)

class AutoTask:
    @classmethod
    def get(self, task, config, seed=42):
        if task in TASK_MAPPING:
            return TASK_MAPPING[task](config, seed)
        raise ValueError(
            "Unrecognized task {} for AutoTask Model: {}.\n"
            "Task name should be one of {}.".format(
                ", ".join(c for c in TASK_MAPPING.keys())
            )
        )
