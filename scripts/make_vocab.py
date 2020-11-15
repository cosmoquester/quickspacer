import argparse
import os
from collections import Counter

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", required=True, help="Directory path containing text files for making vocab")
parser.add_argument("--vocab-path", default="vocab.txt", help="Path to save vocab")

if __name__ == "__main__":
    args = parser.parse_args()

    files = [fname for fname in os.listdir(args.input_dir) if fname.endswith(".txt")]
    print(f"Found {len(files)} files")

    counter = Counter()
    for file_name in tqdm(files):
        with open(os.path.join(args.input_dir, file_name)) as f:
            for line in f:
                counter.update(line.strip())
    print("Counted all files")

    characters = ["<PAD>", "<UNK>"] + [char for char, count in counter.most_common()]

    with open(args.vocab_path, "w") as fout:
        for char in characters:
            fout.write(char + "\n")
    print(f"Saved to {args.vocab_path}")
