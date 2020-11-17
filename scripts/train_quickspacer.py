import argparse
import glob
import json
import os
import shutil

import tensorflow as tf
import tensorflow_addons as tfa

from quickspacer import model
from quickspacer.constant import DEFAULT_VOCAB_PATH
from quickspacer.data import get_dataset
from quickspacer.utils import f1_loss, f1_score, learning_rate_scheduler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="ConvSpacer1")
    parser.add_argument("--model-config-file", type=str, default="configs/conv1-spacer-config.json")
    parser.add_argument("--pretrained-model-path", type=str, default=None)
    parser.add_argument("--min-space-remove-rate", type=float, default=0.3)
    parser.add_argument("--max-space-remove-rate", type=float, default=0.9)
    parser.add_argument("--shuffle-buffer-size", type=int, default=100000)
    parser.add_argument("--vocab-file-path", type=str, default=DEFAULT_VOCAB_PATH, help="vocab file path")
    parser.add_argument("--num-parallel-reads", type=int, default=4)
    parser.add_argument("--num-parallel-calls", type=int, default=4)
    parser.add_argument("--output-path", default="quickspacer_checkpoints/")
    parser.add_argument("--dataset-file-path", default="drama_all.txt", help="a text file or multiple files ex) *.txt")
    parser.add_argument("--loss", type=str, choices=["f1", "bce"], default="bce", help="loss function for training")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--min-learning-rate", type=float, default=1e-8)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--val-batch-size", type=int, default=8192)
    parser.add_argument("--num-val-batch", type=int, default=30000)
    parser.add_argument("--tensorboard-update-freq", type=int, default=100)
    args = parser.parse_args()

    # Copy config file
    os.makedirs(args.output_path)
    with open(os.path.join(args.output_path, "configs.txt"), "w") as fout:
        for k, v in vars(args).items():
            fout.write(f"{k}: {v}\n")
    shutil.copy(args.model_config_file, args.output_path)

    # Construct Dataset
    dataset = get_dataset(
        glob.glob(args.dataset_file_path),
        args.min_space_remove_rate,
        args.max_space_remove_rate,
        args.num_parallel_reads,
        args.num_parallel_calls,
        args.vocab_file_path,
    ).shuffle(args.shuffle_buffer_size)
    train_dataset = dataset.skip(args.num_val_batch).padded_batch(args.batch_size).repeat()
    valid_dataset = dataset.take(args.num_val_batch).padded_batch(max(args.batch_size, args.val_batch_size))

    # Model Initialize
    with open(args.model_config_file) as f:
        model = getattr(model, args.model_name)(**json.load(f))

    # Load pretrained model
    if args.pretrained_model_path:
        model.load_weights(args.pretrained_model_path)
        print("Loaded weights of model")

    # Model Compile
    model.compile(
        optimizer=tfa.optimizers.AdamW(args.weight_decay, args.learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy() if args.loss == "bce" else f1_loss,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Precision(0.5, name="precision"),
            tf.keras.metrics.Recall(0.5, name="recall"),
            f1_score,
        ],
    )

    # Training
    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(args.output_path, "models", "spacer-{epoch}epoch-{val_f1_score:.4f}f1.ckpt"),
                save_weights_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.TensorBoard(
                os.path.join(args.output_path, "logs"), update_freq=args.tensorboard_update_freq
            ),
            tf.keras.callbacks.LearningRateScheduler(
                learning_rate_scheduler(args.epochs, args.learning_rate, args.min_learning_rate), verbose=1
            ),
        ],
    )
