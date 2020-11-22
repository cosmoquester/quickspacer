import argparse
import os
from time import time

import tensorflow as tf

from quickspacer.constant import DEFAULT_MODEL_PATH
from quickspacer.spacer import Spacer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved-model-dir", type=str, default=os.path.join(DEFAULT_MODEL_PATH, "1"))
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--encoding", type=str, default="utf-8")
    args = parser.parse_args()

    spacer = Spacer(saved_model_dir=args.saved_model_dir)
    with open(args.input_path, encoding=args.encoding) as f:
        sentences = f.readlines()
    print(f"[*] Loaded file from {args.input_path}")

    start_time = time()
    spaced_sentences = spacer.space(sentences)
    elapsed_time = time() - start_time
    print("[*] Finished Inferece")
    print(f"[*] Elaspsed Time: {elapsed_time:.8f} seconds")

    if args.output_path:
        with open(args.output_path, "w") as fout:
            fout.writelines(spaced_sentences)
        print(f"[*] Saved spaced file to '{args.output_path}'")
