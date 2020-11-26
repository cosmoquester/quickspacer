import os

import pytest
import tensorflow as tf

from quickspacer.constant import (
    DEFAULT_MODEL_PATH,
    DEFAULT_OOV_INDEX,
    DEFAULT_SPACE_INDEX,
    DEFAULT_VOCAB_PATH,
    DEFAULT_VOCAB_SIZE,
    RESOURCE_PATH,
)
from quickspacer.data import load_vocab


def test_exist_files():
    assert os.path.exists(RESOURCE_PATH)
    assert os.path.exists(DEFAULT_VOCAB_PATH)
    assert os.path.exists(DEFAULT_MODEL_PATH)


def test_vocab():
    vocab = load_vocab(DEFAULT_VOCAB_PATH, DEFAULT_OOV_INDEX)
    assert tf.cast(vocab.lookup(tf.constant(" ")), tf.int32) == DEFAULT_SPACE_INDEX
    assert tf.cast(vocab.lookup(tf.constant("이건절대vocab에없는말")), tf.int32) == DEFAULT_OOV_INDEX
    assert vocab.size() == DEFAULT_VOCAB_SIZE
