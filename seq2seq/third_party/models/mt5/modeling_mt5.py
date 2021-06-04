""" PyTorch mT5 model. """

from transformers.utils import logging
from ..t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Model, T5EncoderModel
from .configuration_mt5 import MT5Config


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "T5Config"
_TOKENIZER_FOR_DOC = "T5Tokenizer"



class MT5Model(T5Model):
    r"""
    This class overrides :class:`~transformers.T5Model`. Please check the superclass for the appropriate documentation
    alongside usage examples.

    Examples::
        >>> from transformers import MT5Model, T5Tokenizer
        >>> model = MT5Model.from_pretrained("google/mt5-small")
        >>> tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
        >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
        >>> summary = "Weiter Verhandlung in Syrien."
        >>> batch = tokenizer.prepare_seq2seq_batch(src_texts=[article], tgt_texts=[summary], return_tensors="pt")
        >>> outputs = model(input_ids=batch.input_ids, decoder_input_ids=batch.labels)
        >>> hidden_states = outputs.last_hidden_state
    """
    model_type = "mt5"
    config_class = MT5Config
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]
    _keys_to_ignore_on_save = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
    ]


class MT5ForConditionalGeneration(T5ForConditionalGeneration):
    r"""
    This class overrides :class:`~transformers.T5ForConditionalGeneration`. Please check the superclass for the
    appropriate documentation alongside usage examples.

    Examples::
        >>> from transformers import MT5ForConditionalGeneration, T5Tokenizer
        >>> model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
        >>> tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
        >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
        >>> summary = "Weiter Verhandlung in Syrien."
        >>> batch = tokenizer.prepare_seq2seq_batch(src_texts=[article], tgt_texts=[summary], return_tensors="pt")
        >>> outputs = model(**batch)
        >>> loss = outputs.loss
    """

    model_type = "mt5"
    config_class = MT5Config
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
    ]
    _keys_to_ignore_on_save = [
        r"encoder\.embed_tokens\.weight",
    ]


class MT5EncoderModel(T5EncoderModel):
    r"""
    This class overrides :class:`~transformers.T5EncoderModel`. Please check the superclass for the appropriate
    documentation alongside usage examples.

    Examples::

        >>> from transformers import MT5EncoderModel, T5Tokenizer
        >>> model = MT5EncoderModel.from_pretrained("google/mt5-small")
        >>> tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
        >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
        >>> input_ids = tokenizer(article, return_tensors="pt").input_ids
        >>> outputs = model(input_ids)
        >>> hidden_state = outputs.last_hidden_state
    """

    model_type = "mt5"
    config_class = MT5Config
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
    ]
    _keys_to_ignore_on_save = [
        r"encoder\.embed_tokens\.weight",
    ]
