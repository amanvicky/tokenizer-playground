#!/usr/bin/env python3
"""Train a byte-level BPE tokenizer on a text corpus.

Byte-level, no word pre-tokenization (GPT-2 splits words first; this toy
keeps the algorithm bare so the whole thing is one readable loop). Emits
merges.json consumed by both the Python tests and the browser encoder:

    {"model": "bpe", "base_vocab": 256, "merges": [[a, b], ...]}

ids 0..255 are raw bytes; merge k creates id 256+k by joining a and b.
"""

import argparse
import json


def train(text, n_merges):
    ids = list(text.encode("utf-8"))
    merges = []
    for _ in range(n_merges):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        if not counts:
            break
        best, _ = max(counts.items(), key=lambda kv: kv[1])
        new_id = 256 + len(merges)
        merged, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best:
                merged.append(new_id)
                i += 2
            else:
                merged.append(ids[i])
                i += 1
        ids = merged
        merges.append(list(best))
    return merges


def encode(text, merges):
    """Apply merges in order of rank — the same loop the browser runs.
    Accepts merges as lists (train's output) or tuples."""
    ids = list(text.encode("utf-8"))
    ranks = {tuple(pair): rank for rank, pair in enumerate(merges)}
    while True:
        best_rank, best_idx = None, None
        for i, pair in enumerate(zip(ids, ids[1:])):
            r = ranks.get(pair)
            if r is not None and (best_rank is None or r < best_rank):
                best_rank, best_idx = r, i
        if best_rank is None:
            return ids
        pair = tuple(merges[best_rank])  # tuple, not list: list == tuple is always False
        merged, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                merged.append(256 + best_rank)
                i += 2
            else:
                merged.append(ids[i])
                i += 1
        ids = merged


def decode(ids, merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for rank, (a, b) in enumerate(merges):
        vocab[256 + rank] = vocab[a] + vocab[b]
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


def main():
    p = argparse.ArgumentParser(description="train byte-level BPE")
    p.add_argument("corpus")
    p.add_argument("--merges", type=int, default=400)
    p.add_argument("--out", default="merges.json")
    args = p.parse_args()

    text = open(args.corpus, encoding="utf-8").read()
    merges = train(text, args.merges)
    json.dump({"model": "bpe", "base_vocab": 256, "merges": merges},
              open(args.out, "w", encoding="utf-8"), indent=1)
    ids = encode(text, merges)
    print(f"corpus {len(text):,} chars -> {len(merges)} merges, "
          f"{len(ids):,} tokens ({len(text) / len(ids):.1f} chars/token)")


if __name__ == "__main__":
    main()
