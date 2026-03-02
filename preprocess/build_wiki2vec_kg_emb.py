#!/usr/bin/env python
import argparse
import numpy as np
import torch

from fairseq.data import Dictionary
from wikipedia2vec import Wikipedia2Vec


def build_kg_matrix(dict_path, wiki2vec_path, out_path, unk_init="zero"):
    print(f"Loading Dictionary from {dict_path}")
    dictionary = Dictionary.load(dict_path)
    vocab_size = len(dictionary)
    pad_idx = dictionary.pad()
    print(f"Vocab size = {vocab_size}, pad_idx = {pad_idx}")

    print(f"Loading Wikipedia2Vec model from {wiki2vec_path}")
    wiki2vec = Wikipedia2Vec.load(wiki2vec_path)

    # Probe embedding dim with a known word (fallback to first vocab word)
    try:
        sample_vec = wiki2vec.get_word_vector("Japan")
    except KeyError:
        # Safety fallback: pick any token from word_vocab
        first_word = next(iter(wiki2vec.dictionary.word2id.keys()))
        sample_vec = wiki2vec.get_word_vector(first_word)
        print(f"[KG] Japan not in word vocab, fallback to {first_word} to probe dimension")

    dim = sample_vec.shape[0]
    print(f"KG embedding dim = {dim}")

    kg_matrix = np.zeros((vocab_size, dim), dtype=np.float32)

    hit_word = 0
    hit_entity = 0
    miss = 0

    for idx in range(vocab_size):
        token = dictionary[idx]      
        # Use zero vector for special symbols
        if idx == pad_idx or token in {"<s>", "</s>", "<unk>"}:
            continue

        surface = token.replace("▁", "")
        vec = None

        # 1) Try as a word
        try:
            vec = wiki2vec.get_word_vector(surface)
            hit_word += 1
        except KeyError:
            pass

        # 2) Then try as an entity
        if vec is None:
            try:
                vec = wiki2vec.get_entity_vector(surface)
                hit_entity += 1
            except KeyError:
                pass

        if vec is None:
            miss += 1
            if unk_init == "randn":
                kg_matrix[idx] = np.random.normal(scale=0.01, size=(dim,))
        else:
            kg_matrix[idx] = vec

        if idx % 1000 == 0:
            print(f"  processed {idx}/{vocab_size}")

    print(f"Done. word hits = {hit_word}, entity hits = {hit_entity}, miss = {miss}")
    kg_tensor = torch.from_numpy(kg_matrix)
    print(f"Saving to {out_path}")
    torch.save(kg_tensor, out_path)
    print("Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dict", required=True, help="path to dict.ja.txt")
    parser.add_argument("--wiki2vec", required=True, help="path to jawiki_*.pkl")
    parser.add_argument("--out", required=True, help="output .pt file")
    parser.add_argument("--unk-init", default="zero", choices=["zero", "randn"])
    args = parser.parse_args()

    build_kg_matrix(args.dict, args.wiki2vec, args.out, args.unk_init)