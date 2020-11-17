import argparse
import json
from functools import partial

import tensorflow as tf

from quickspacer import model
from quickspacer.constant import DEFAULT_VOCAB_PATH


def sentence_to_index(
    sentence: tf.Tensor, vocab: tf.keras.layers.experimental.preprocessing.TextVectorization,
):
    """
    functions as same as quickspacer.data.sentence_to_index.
    But It use TextVectorization layer.
    """
    mapped = vocab(tf.strings.unicode_split(sentence, "UTF-8"))

    # 'TextVectorization' has default tokens of "", "[UNK]" which are not used but not removable
    # So for correct mapping, should minus 2
    mapped = mapped - 2
    return tf.cast(mapped, tf.int32)


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="ConvSpacer1", help="Model class name in quickspacer.model")
    parser.add_argument("--model-config-file", type=str, default="configs/conv1-spacer-config.json", help="Config file path for model")
    parser.add_argument("--model-weight-path", type=str, required=True, help="Model weight file path saved in training")
    parser.add_argument("--vocab-path", type=str, default=DEFAULT_VOCAB_PATH, help="Vocab file path")
    parser.add_argument("--output-path", type=str, default="saved_spacer_model/1", help="Savedmodel path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold to space")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    # fmt: on

    with open(args.model_config_file) as f:
        model = getattr(model, args.model_name)(**json.load(f))
    model.load_weights(args.model_weight_path)
    print("Loaded weights of model")

    with open(args.vocab_path) as f:
        data = f.read().strip().split("\n")
    vocab = tf.keras.layers.experimental.preprocessing.TextVectorization(trainable=False, standardize=None, split=None)
    vocab.set_vocabulary(data)
    print("Loaded vocab")

    threshold = tf.constant(args.threshold)
    batch_size = tf.constant(args.batch_size, dtype=tf.int64)

    @tf.function(input_signature=[tf.TensorSpec((None,), tf.string)])
    def space_texts(texts):
        # Construct dataset
        tokens = tf.map_fn(
            partial(sentence_to_index, vocab=vocab),
            texts,
            fn_output_signature=tf.RaggedTensorSpec([None], tf.int32, ragged_rank=0),
        )
        dataset = tf.data.Dataset.from_tensor_slices(tokens)
        dataset = dataset.map(lambda x: x)  # For using padded_batch from dataset made of RaggedTensor
        dataset = dataset.padded_batch(batch_size)

        # Inference
        space_mask = tf.constant((), dtype=tf.bool)
        for data in dataset:
            output = model(data)
            space_mask = tf.reshape(space_mask, (-1, tf.shape(output)[1]))
            space_mask = tf.concat((space_mask, output < threshold), 0)

        # Make spaced sentence
        texts = tf.strings.unicode_split(texts, "UTF-8")
        spaced_sentences = tf.where(space_mask, texts.to_tensor(), (texts + " ").to_tensor())
        spaced_sentences = tf.strings.reduce_join(spaced_sentences, axis=1)

        return {"spaced_sentences": spaced_sentences}

    @tf.function(input_signature=[tf.TensorSpec((None, None), tf.int32)])
    def model_inference(tokens):
        outputs = model(tokens)
        return outputs

    model.vocab = vocab
    tf.saved_model.save(
        model, args.output_path, signatures={"serving_default": space_texts, "model_inference": model_inference},
    )
