import os

import tensorflow as tf

DEFAULT_VOCAB_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "resources", "vocab.txt"))
DEFAULT_OOV_INDEX = tf.constant(1, dtype=tf.int32)
DEFAULT_SPACE_INDEX = tf.constant(2, dtype=tf.int32)
with open(DEFAULT_VOCAB_PATH) as f:
    DEFAULT_VOCAB_SIZE = len(list(f))
