"""Tests for the BPE trainer: merge correctness, determinism, roundtrip."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "trainer"))
import train_bpe as tb


def test_first_merge_is_the_most_frequent_pair():
    merges = tb.train("aaabdaaabac", 1)
    # 'aa' occurs 4 times, more than any other adjacent pair
    assert merges == [[97, 97]]


def test_merges_are_deterministic():
    a = tb.train("the cat sat on the mat. the end.", 50)
    b = tb.train("the cat sat on the mat. the end.", 50)
    assert a == b


def test_merge_ids_are_contiguous_from_256():
    merges = tb.train("hello hello hello world", 30)
    for rank, pair in enumerate(merges):
        assert len(pair) == 2
        assert all(0 <= i < 256 + rank for i in pair)   # can only use what exists


def test_roundtrip_on_tricky_text():
    text = ("Attention is all you need! ₹700 chai, déjà vu, 🚀 launch, "
            "def train(step): return step * 2\nमुंबई\ttab\tтекст")
    merges = tb.train(text * 3, 400)
    assert tb.decode(tb.encode(text, merges), merges) == text


def test_encoding_shrinks_repeated_text():
    text = "the quick brown fox " * 20
    merges = tb.train(text, 300)
    ids = tb.encode(text, merges)
    assert len(ids) < len(text.encode("utf-8")) / 2


def test_encode_empty_and_single():
    merges = tb.train("abc abc abc", 10)
    assert tb.encode("", merges) == []
    assert tb.decode([], merges) == ""
    assert tb.decode(tb.encode("z", merges), merges) == "z"


def test_encode_terminates_on_raw_train_output():
    """Regression: train() returns merges as lists, and tuple-vs-list comparison
    silently made encode loop forever. The raw output must work as-is."""
    text = "the quick brown fox " * 10
    merges = tb.train(text, 100)
    ids = tb.encode(text, merges)
    assert tb.decode(ids, merges) == text


def test_merges_json_shape(tmp_path):
    corpus = tmp_path / "c.txt"
    corpus.write_text("banana bandana", encoding="utf-8")
    merges = tb.train(corpus.read_text(), 8)
    doc = {"model": "bpe", "base_vocab": 256, "merges": merges}
    assert doc["base_vocab"] == 256
    assert all(isinstance(v, int) for pair in merges for v in pair)


def test_shipped_merges_roundtrip_on_samples():
    """The merges.json that ships to the browser must roundtrip real text."""
    root = Path(__file__).parent.parent
    doc = json.loads((root / "merges.json").read_text())
    merges = [tuple(p) for p in doc["merges"]]
    for sample in [
        "Attention is the idea that made modern AI possible.",
        "def train_bpe(text, n_merges): return merges",
        "chai at seven rupees, laptop bag on the seat 🚀",
    ]:
        assert tb.decode(tb.encode(sample, merges), merges) == sample
