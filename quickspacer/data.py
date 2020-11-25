import random
from functools import partial
from typing import Tuple, Union

import tensorflow as tf

from .constant import DEFAULT_OOV_INDEX, DEFAULT_SPACE_INDEX, DEFAULT_VOCAB_PATH


def get_dataset(
    dataset_file_path: str,
    min_remove_rate: float,
    max_remove_rate: float,
    num_parallel_reads: int = 4,
    num_parallel_calls: int = 4,
    vocab_file_path: str = DEFAULT_VOCAB_PATH,
    oov_index: Union[tf.Tensor, int] = DEFAULT_OOV_INDEX,
    space_index: Union[tf.Tensor, int] = DEFAULT_SPACE_INDEX,
) -> tf.data.Dataset:
    """
    Read dataset file and construct tensorflow dataset

    :param dataset_file_path: dataset (txt) file path.
    :param min_remove_rate: minimum rate to remove spaces of all spaces.
    :param max_remove_rate: maximum rate to remove spaces of all spaces.
    :param num_parallel_reads: number for parallel reading
    :param num_parallel_calls: number for parallel mapping
    :param vocab_file_path: vocab file path.
    :param oov_index: OOV index.
    :param space_index: " " character index in vocab.
    """
    vocab = load_vocab(vocab_file_path, oov_index)
    dataset = (
        tf.data.TextLineDataset(dataset_file_path, num_parallel_reads=num_parallel_reads)
        .map(partial(sentence_to_index, vocab=vocab, oov_index=oov_index), num_parallel_calls=num_parallel_calls)
        .map(
            partial(
                sentence_to_dataset,
                min_remove_rate=min_remove_rate,
                max_remove_rate=max_remove_rate,
                space_index=space_index,
            ),
            num_parallel_calls=num_parallel_calls,
        )
    )
    return dataset


def sentence_to_index(sentence: tf.Tensor, vocab: tf.lookup.StaticHashTable) -> tf.Tensor:
    """
    Mapping character to number with vocab

    :param sentence: string type 0D (scalar) tensor of sentence.
    :param vocab: vocab loaded using 'load_vocab'.
    """
    mapped = vocab.lookup(tf.strings.unicode_split(sentence, "UTF-8"))
    return tf.cast(mapped, tf.int32)


def sentence_to_dataset(
    token_ords: tf.Tensor,
    min_remove_rate: Union[float, tf.Tensor],
    max_remove_rate: Union[float, tf.Tensor],
    space_index=DEFAULT_SPACE_INDEX,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Remove spaces in tokens by remove_rate

    :param token_ords: int type 1D tensor of a sentence shaped (SequenceLength).
    :param min_remove_rate: minimum rate to remove spaces of all spaces.
    :param max_remove_rate: maximum rate to remove spaces of all spaces.
    :param space_index: vocab index of space " ".
    :return: tuple of input and label. Both shapes are (SequenceLengthAfterRemoveSpace)
    """
    remove_rate = tf.random.uniform((), min_remove_rate, max_remove_rate)
    space_indices = tf.where(token_ords == space_index)[:, 0]
    num_removed_spaces = tf.cast(tf.math.ceil(tf.cast(tf.size(space_indices), tf.float32) * remove_rate), tf.int32)
    removed_space_indices = tf.random.shuffle(space_indices)[:num_removed_spaces]
    token_ords = tf.tensor_scatter_nd_update(
        token_ords, removed_space_indices[:, tf.newaxis], tf.fill(tf.shape(removed_space_indices), -1)
    )
    label_indices = tf.sort(removed_space_indices) - tf.range(1, tf.size(removed_space_indices) + 1, dtype=tf.int64)
    labels = tf.scatter_nd(
        label_indices[:, tf.newaxis], tf.ones(num_removed_spaces, tf.int32), [tf.size(token_ords) - num_removed_spaces]
    )
    # In case space is consecutive (ex "안   녕") and remove consecutive space
    # labels can be value more than 1, so prevent that situation
    labels = tf.where(labels <= 1, labels, 1)

    return tf.gather(token_ords, tf.where(token_ords != -1))[:, 0], labels


def load_vocab(vocab_file_path: str, oov_index: Union[int, tf.Tensor]) -> tf.lookup.StaticHashTable:
    """
    Load vocab from file
    """
    if isinstance(oov_index, int):
        oov_index = tf.constant(oov_index, tf.int64)
    if isinstance(oov_index, tf.Tensor):
        oov_index = tf.cast(oov_index, tf.int64)

    vocab = tf.lookup.StaticHashTable(
        tf.lookup.TextFileInitializer(
            vocab_file_path,
            key_dtype=tf.string,
            key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
            value_dtype=tf.int64,
            value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
            delimiter="\n",
        ),
        default_value=oov_index,
    )
    return vocab
