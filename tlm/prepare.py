import os
import sys
import pickle
import argparse
import numpy as np
from tokenizer import Tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-train", type=str, required=True)
    parser.add_argument("--input-val", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="data/openwebtext")
    parser.add_argument("--vocab-size", type=int, default=50304)
    parser.add_argument("--model-prefix", type=str, default="tokenizer")
    args = parser.parse_args()

    if not os.path.isfile(args.input_train):
        print(f"Error: training file '{args.input_train}' not found.")
        sys.exit(1)
    if not os.path.isfile(args.input_val):
        print(f"Error: validation file '{args.input_val}' not found.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input_train, 'r', encoding='utf-8') as f:
        train_text = f.read()

    tokenizer = Tokenizer()
    tokenizer.train(train_text, args.vocab_size)

    model_path = os.path.join(args.output_dir, args.model_prefix)
    tokenizer.save(model_path)

    train_ids = tokenizer.encode(train_text)
    del train_text

    with open(args.input_val, 'r', encoding='utf-8') as f:
        val_text = f.read()
    val_ids = tokenizer.encode(val_text)
    del val_text

    max_id = max(max(train_ids), max(val_ids))
    if max_id < 2**16:
        dtype = np.uint16
    else:
        dtype = np.uint32

    train_bin_path = os.path.join(args.output_dir, 'train.bin')
    train_arr = np.array(train_ids, dtype=dtype)
    train_arr.tofile(train_bin_path)

    val_bin_path = os.path.join(args.output_dir, 'val.bin')
    val_arr = np.array(val_ids, dtype=dtype)
    val_arr.tofile(val_bin_path)

    meta = {
        'vocab_size': args.vocab_size,
        'itos': None,
    }
    meta_path = os.path.join(args.output_dir, 'meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

if __name__ == "__main__":
    main()