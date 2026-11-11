# tokenizer-playground

[![CI](https://github.com/amanvicky/tokenizer-playground/actions/workflows/ci.yml/badge.svg)](https://github.com/amanvicky/tokenizer-playground/actions/workflows/ci.yml)

**Watch a BPE tokenizer chop your text into pieces, live in your browser:** [amanvicky.github.io/tokenizer-playground](https://amanvicky.github.io/tokenizer-playground/)

Byte-level BPE — the algorithm GPT-style models use to turn text into numbers — starting from 256 raw-byte tokens and repeatedly merging the most frequent adjacent pair. The merge list here was learned offline by a ~60-line Python trainer ([`trainer/train_bpe.py`](trainer/train_bpe.py)) on this repo's own corpus; your browser applies the exact same list as you type.

## What you see

- **Colored token chips** as you type — hover one for its token id and raw bytes
- **The stat that decides your API bill**: tokens, chars, bytes, chars-per-token
- **"What it learned first"** — the first merges, literally `e + space → e `, `t + h → th`
- **A roundtrip badge** — `decode(encode(text)) === text`, recomputed on every keystroke
- Samples: plain English, Python code, and emoji + Devanagari (watch ₹ and 🚀 shatter into four bytes each)

## How it's verified

Two independent implementations must agree:

- **`parity-check.mjs`** — a Node script that runs the browser's encoder against the Python trainer's on plain-English, code, and unicode samples; every token id must match
- **9 pytest tests** on the trainer: merge correctness on the classic `aaabdaaabac` example, determinism, contiguous ids, roundtrip on emoji/Devanagari text, and a regression for a real bug the suite caught (a `tuple`-vs-`list` comparison that made `encode` loop forever — the stack trace found it in one pass)

```bash
pytest tests/ -q          # trainer tests
node parity-check.mjs     # JS == Python, token by token
python trainer/train_bpe.py corpus.txt --merges 400 --out merges.json   # retrain
```

## Honest scope

GPT-2 tokenizers split words with a regex first and use ~50K merges; this toy skips the word-splitting (merges may cross spaces — visible as `e·` chips) and trains 400 merges in seconds. The algorithm is the same; the scale is not.

The corpus is the repo's own text — paragraphs about attention, fine-tuning, Hyderabad traffic and code review — so the tokenizer learned English the way the README talks about it.
