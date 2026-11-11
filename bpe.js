/* Byte-level BPE encode/decode — mirrors trainer/train_bpe.py exactly.
   Pure module: no DOM, so Node can run it against the Python implementation.
   ids 0..255 are raw bytes; merge k creates id 256+k from pair [a, b]. */

let MERGES = [];
let RANKS = new Map();
let VOCAB = new Map();

export function setMerges(doc) {
  MERGES = doc.merges;
  RANKS = new Map(MERGES.map((pair, rank) => [pair[0] + "," + pair[1], rank]));
  VOCAB = new Map();
  for (let i = 0; i < 256; i++) VOCAB.set(i, [i]);
  MERGES.forEach(([a, b], rank) => VOCAB.set(256 + rank, [...VOCAB.get(a), ...VOCAB.get(b)]));
}

export function mergeCount() {
  return MERGES.length;
}

export function mergePairs() {
  return MERGES;
}

export function encode(text) {
  let ids = Array.from(new TextEncoder().encode(text));
  for (;;) {
    // lowest-rank adjacent pair wins; replace every occurrence in one pass
    let bestRank = Infinity;
    for (let i = 0; i < ids.length - 1; i++) {
      const r = RANKS.get(ids[i] + "," + ids[i + 1]);
      if (r !== undefined && r < bestRank) bestRank = r;
    }
    if (bestRank === Infinity) return ids;
    const out = [];
    for (let i = 0; i < ids.length; i++) {
      if (i < ids.length - 1 && RANKS.get(ids[i] + "," + ids[i + 1]) === bestRank) {
        out.push(256 + bestRank);
        i += 1;
      } else {
        out.push(ids[i]);
      }
    }
    ids = out;
  }
}

export function decode(ids) {
  const bytes = new Uint8Array(ids.flatMap(id => VOCAB.get(id)));
  return new TextDecoder("utf-8", { fatal: false }).decode(bytes);
}

export function tokenBytes(id) {
  return VOCAB.get(id);
}
