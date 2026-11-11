/* UI for the tokenizer playground: textarea, live token chips, stats, vocab panel.
   The tokenizer itself lives in bpe.js (pure, Node-testable). */

import { setMerges, encode, decode, mergeCount, tokenBytes, mergePairs } from "./bpe.js";

const SAMPLES = {
  simple: "Attention is all you need. The tokenizer turns your sentence into pieces a model can read, and the number of pieces is the number your API bill cares about.",
  code: "def train(text, n_merges):\n    ids = list(text.encode('utf-8'))\n    for _ in range(n_merges):\n        counts = count_pairs(ids)\n        best = max(counts, key=counts.get)\n        ids = merge(ids, best, 256 + len(merges))\n    return ids",
  weird: "chai at ₹7, déjà vu, 🚀 launch, मुंबई — typography and emoji are just bytes: 0xF0 0x9F 0x9A 0x80",
};

function visible(text) {
  return text.replace(/ /g, "\u00B7").replace(/\n/g, "\u21B5").replace(/\t/g, "\u21E5");
}

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function showBytes(bytes) {
  try {
    return new TextDecoder("utf-8", { fatal: false }).decode(new Uint8Array(bytes));
  } catch (e) {
    return "?";
  }
}

function render(text) {
  const ids = encode(text);
  const chips = document.getElementById("chips");
  chips.innerHTML = "";
  const COLORS = 7;
  ids.forEach((id, i) => {
    const span = document.createElement("span");
    span.className = "tok c" + (i % COLORS);
    span.textContent = visible(showBytes(tokenBytes(id)));
    span.title = "token " + id + "  bytes: " + tokenBytes(id).join(" ");
    chips.appendChild(span);
  });

  const byteLen = new TextEncoder().encode(text).length;
  document.getElementById("stats").innerHTML =
    `<b>${ids.length.toLocaleString()}</b> tokens` +
    ` · ${text.length.toLocaleString()} chars` +
    ` · ${byteLen.toLocaleString()} bytes` +
    ` · ${text.length ? (text.length / ids.length).toFixed(2) : "—"} chars/token`;

  const back = decode(ids);
  const rt = document.getElementById("roundtrip");
  rt.textContent = back === text ? "roundtrip ok: decode(encode(text)) === text" : "roundtrip mismatch!";
  rt.className = "roundtrip " + (back === text ? "ok" : "bad");

  document.getElementById("vocabInfo").innerHTML =
    `base bytes <b>256</b><br>learned merges <b>${mergeCount().toLocaleString()}</b><br>` +
    `total vocab <b>${(256 + mergeCount()).toLocaleString()}</b>`;
}

function renderTopMerges() {
  const box = document.getElementById("topMerges");
  box.innerHTML = "";
  const show = (id) => visible(showBytes(tokenBytes(id)));
  mergePairs().slice(0, 14).forEach((pair, rank) => {
    const row = document.createElement("div");
    row.className = "merge";
    row.innerHTML = `<span class="rank">${rank + 1}</span> ` +
      `<code>${escapeHtml(show(pair[0]))}</code> + <code>${escapeHtml(show(pair[1]))}</code> ` +
      `→ <code>${escapeHtml(show(256 + rank))}</code>`;
    box.appendChild(row);
  });
}

const input = document.getElementById("input");
input.addEventListener("input", () => render(input.value));
document.querySelectorAll("button[data-sample]").forEach(btn => {
  btn.addEventListener("click", () => {
    input.value = SAMPLES[btn.dataset.sample];
    render(input.value);
  });
});

(async function start() {
  try {
    const res = await fetch("merges.json");
    setMerges(await res.json());
    renderTopMerges();
    input.value = SAMPLES.simple;
    render(input.value);
  } catch (e) {
    document.getElementById("chips").textContent = "couldn't load merges.json — " + e;
  }
})();
