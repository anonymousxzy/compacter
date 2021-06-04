""" BERT model configuration """
from transformers.models.bert import BertConfig

class BertConfig(BertConfig):
    def __init__(self, train_lang_adapters=False,
                 train_task_adapters=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.train_task_adapters = train_task_adapters
        self.train_lang_adapters = train_lang_adapters

