/* Node parity check: the browser encoder must produce the exact token ids
   the Python trainer produces. Fails loudly if the two drift apart. */

import { readFileSync, writeFileSync } from "node:fs";
import { setMerges, encode, decode } from "./bpe.js";
import { execFileSync } from "node:child_process";

const doc = JSON.parse(readFileSync("merges.json", "utf-8"));
setMerges(doc);

const samples = [
  "Attention is all you need. The tokenizer turns your sentence into pieces a model can read.",
  "def train(text, n_merges):\n    return merge(ids, best, 256 + len(merges))",
  "chai at \u20B97, d\u00E9j\u00E0 vu, \u{1F680} launch, \u092E\u0941\u0902\u092C\u0908 \u2014 bytes all the way down",
];

const py = `
import json, sys
sys.path.insert(0, "trainer")
import train_bpe as tb
doc = json.load(open("merges.json", encoding="utf-8"))
merges = [tuple(p) for p in doc["merges"]]
samples = json.load(open("parity-samples.json", encoding="utf-8"))
print(json.dumps([tb.encode(s, merges) for s in samples]))
`;

writeFileSync("parity-samples.json", JSON.stringify(samples), "utf-8");
const pyOut = execFileSync("python", ["-X", "utf8", "-c", py], { encoding: "utf-8" });
const pyIds = JSON.parse(pyOut.trim().splitlines ? pyOut.trim() : pyOut.trim());

let fail = 0;
samples.forEach((s, i) => {
  const jsIds = encode(s);
  const roundtrip = decode(jsIds) === s;
  const match = JSON.stringify(jsIds) === JSON.stringify(pyIds[i]);
  console.log(`sample ${i}: ${jsIds.length} tokens | roundtrip ${roundtrip ? "ok" : "FAIL"} | js==py ${match ? "ok" : "FAIL"}`);
  if (!match || !roundtrip) fail += 1;
});
process.exit(fail ? 1 : 0);
