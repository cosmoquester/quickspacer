from typing import Iterable

import tensorflow as tf

from .constant import DEFAULT_MODEL_PATH


class Spacer:
    def __init__(self, saved_model_dir=DEFAULT_MODEL_PATH):
        self.model = tf.saved_model.load(saved_model_dir)

    def space(self, sentences: Iterable[str]) -> Iterable[str]:
        sentences = tf.constant(sentences)
        spaced_sentences = self.model.signatures["serving_default"](sentences)["spaced_sentences"]
        spaced_sentences = [sentence.numpy().decode("utf-8") for sentence in spaced_sentences]
        return spaced_sentences
