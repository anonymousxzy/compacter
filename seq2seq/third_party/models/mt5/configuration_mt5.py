""" mT5 model configuration """
from transformers.models.mt5 import MT5Config

class MT5Config(MT5Config):
    def __init__(self, train_lang_adapters=False,
                 train_task_adapters=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.train_task_adapters = train_task_adapters
        self.train_lang_adapters = train_lang_adapters