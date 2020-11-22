import os

import tensorflow as tf

RESOURCE_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), "resources"))
DEFAULT_VOCAB_PATH = os.path.join(RESOURCE_PATH, "vocab.txt")
DEFAULT_MODEL_PATH = os.path.join(RESOURCE_PATH, "default_saved_model")
DEFAULT_OOV_INDEX = tf.constant(1, dtype=tf.int32)
DEFAULT_SPACE_INDEX = tf.constant(2, dtype=tf.int32)
with open(DEFAULT_VOCAB_PATH, encoding="utf-8") as f:
    DEFAULT_VOCAB_SIZE = len(list(f))
