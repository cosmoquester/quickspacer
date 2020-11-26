import argparse
import json
from functools import partial

import tensorflow as tf

from quickspacer import model
from quickspacer.constant import DEFAULT_OOV_INDEX, DEFAULT_VOCAB_PATH
from quickspacer.data import load_vocab, sentence_to_index

if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="ConvSpacer1", help="Model class name in quickspacer.model")
    parser.add_argument("--model-config-file", type=str, default="configs/conv1-spacer-config.json", help="Config file path for model")
    parser.add_argument("--model-weight-path", type=str, required=True, help="Model weight file path saved in training")
    parser.add_argument("--vocab-path", type=str, default=DEFAULT_VOCAB_PATH, help="Vocab file path")
    parser.add_argument("--oov-index", type=int, default=DEFAULT_OOV_INDEX, help="OOV Index")
    parser.add_argument("--output-path", type=str, default="saved_spacer_model/1", help="Savedmodel path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold to space")
    args = parser.parse_args()
    # fmt: on

    with open(args.model_config_file) as f:
        model = getattr(model, args.model_name)(**json.load(f))
    model.load_weights(args.model_weight_path)
    print("Loaded weights of model")

    vocab = load_vocab(args.vocab_path, args.oov_index)
    print("Loaded vocab")

    @tf.function(input_signature=[tf.TensorSpec((None,), tf.string)])
    def space_texts(texts):
        # Construct dataset
        tokens = tf.map_fn(
            partial(sentence_to_index, vocab=vocab),
            texts,
            fn_output_signature=tf.RaggedTensorSpec([None], tf.int32, ragged_rank=0),
        ).to_tensor()

        # Model inference
        output = model(tokens)

        # Make spaced sentence
        texts = tf.strings.unicode_split(texts, "UTF-8").to_tensor()
        spaced_sentences = tf.where(output < args.threshold, texts, texts + " ")
        spaced_sentences = tf.strings.strip(tf.strings.reduce_join(spaced_sentences, axis=1))

        return {"spaced_sentences": spaced_sentences}

    @tf.function(input_signature=[tf.TensorSpec((None, None), tf.int32)])
    def model_inference(tokens):
        outputs = model(tokens)
        return outputs

    model.vocab = vocab
    tf.saved_model.save(
        model,
        args.output_path,
        signatures={"serving_default": space_texts, "model_inference": model_inference},
    )
