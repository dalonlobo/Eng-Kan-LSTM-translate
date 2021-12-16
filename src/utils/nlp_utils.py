import re
import unicodedata
from typing import Any

import numpy as np
import tensorflow as tf


def unicode_to_ascii(s: str) -> str:
    "Normalizes to NFD and remove accented characters"
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def preprocess_en(s: str) -> str:
    "Convert to ascii, substitue punctuations and append 'start' and 'end' tokens"
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([?.!|,¿])", r" \1 ", s)
    s = re.sub(r'[" "]+', " ", s)
    s = s.strip()
    s = "<start> " + s + " <end>"
    return s


def preprocess_kn(s: str) -> str:
    "Substitue punctuations and append 'start' and 'end' tokens"
    s = s.lower().strip()
    s = re.sub(r"([?.!|,¿])", r" \1 ", s)
    s = re.sub(r'[" "]+', " ", s)
    s = s.strip()
    s = "<start> " + s + " <end>"
    return s


def tokenize(lang: list) -> Any:
    "Tokenize the sentences"
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters="")
    tokenizer.fit_on_texts(lang)
    tensor = tokenizer.texts_to_sequences(lang)
    return tensor, tokenizer


def get_clean_mtdata(mt_data: list, max_sen_limit: int = 7) -> Any:
    """Return the input and target tensors as well as tokenizers

    Args:
        mt_data (list): dataset in list of list

    Returns:
        Any: input_tensor, input_tokenizer, target_tensor, target_tokenizer
    """
    ip_lang = [preprocess_en(w[0]) for w in mt_data]
    tr_lang = [preprocess_kn(w[1]) for w in mt_data]

    input_tensor, input_tokenizer = tokenize(ip_lang)
    target_tensor, target_tokenizer = tokenize(tr_lang)

    _input_tensor, _target_tensor = [], []
    for idx, val in enumerate(target_tensor):
        if len(val) <= max_sen_limit:
            _input_tensor.append(input_tensor[idx])
            _target_tensor.append(target_tensor[idx])
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(_input_tensor, maxlen=None, padding="post")
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(_target_tensor, maxlen=None, padding="post")

    return input_tensor, input_tokenizer, target_tensor, target_tokenizer


def logits_to_sentence(logits: Any, tokenizer: Any) -> Any:
    """
    Turn logits from a neural network into sentence using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = "<PAD>"

    return " ".join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])
