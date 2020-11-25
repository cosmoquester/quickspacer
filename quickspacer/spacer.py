import os
from typing import Iterable

import tensorflow as tf

from .constant import DEFAULT_MODEL_PATH


class Spacer:
    def __init__(self, saved_model_dir=None, level=1):
        if saved_model_dir is None:
            assert level in (1, 2, 3), "level should be one of (1,2,3)!"
            saved_model_dir = os.path.join(DEFAULT_MODEL_PATH, str(level))

        self.model = tf.saved_model.load(saved_model_dir)

    def space(self, sentences: Iterable[str], batch_size=None) -> Iterable[str]:
        all_spaced_sentences = []

        if batch_size is None:
            batch_size = len(sentences)

        while sentences:
            batch_sentences = tf.constant(sentences[:batch_size])
            spaced_sentences = self.model.signatures["serving_default"](batch_sentences)["spaced_sentences"]
            all_spaced_sentences.extend([sentence.numpy().decode("utf-8") for sentence in spaced_sentences])
            sentences = sentences[batch_size:]
        return all_spaced_sentences
