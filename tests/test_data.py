import pytest
import tensorflow as tf

from quickspacer.constant import DEFAULT_SPACE_INDEX
from quickspacer.data import sentence_to_dataset


@pytest.mark.parametrize(
    "token_ords,remove_rate",
    [
        (tf.strings.unicode_decode("안 녕 하 세 요", "UTF-8"), 0.5),
        (tf.strings.unicode_decode("그만 좀 하자", "UTF-8"), 0.1),
        (tf.strings.unicode_decode("나는 테스트 코드 더 안 짜고 싶다구!", "UTF-8"), 0.2),
        (tf.strings.unicode_decode("으아아 아 아 아ㅏ 아 제바아아알", "UTF-8"), 0.3),
    ],
)
def test_sentence_to_dataset(token_ords, remove_rate):
    num_space = tf.math.count_nonzero(token_ords == DEFAULT_SPACE_INDEX)
    num_removed = tf.cast(tf.math.ceil(tf.cast(num_space, tf.float32) * remove_rate), tf.int32)
    sentence_data, labels = sentence_to_dataset(token_ords, remove_rate)

    tf.debugging.assert_equal(tf.size(token_ords) - tf.size(sentence_data), num_removed)
    tf.debugging.assert_equal(
        num_space - tf.math.count_nonzero(sentence_data == DEFAULT_SPACE_INDEX), tf.cast(num_removed, tf.int64)
    )

    restored_data = []
    for token, label in zip(sentence_data, labels):
        restored_data.append(token)
        if label:
            restored_data.append(DEFAULT_SPACE_INDEX)

    tf.debugging.assert_equal(
        token_ords, tf.concat(restored_data, axis=0),
    )
