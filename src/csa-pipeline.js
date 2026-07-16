/* DeepSeek-V4 Compressed Sparse Attention pipeline explainer.
   The values are deterministic teaching data. The structure mirrors §2.3.1:
   overlapping KV compression → multi-head lightning indexer → top-k plus
   sliding-window KV → shared-KV MQA → grouped output projection. */
(function () {
  "use strict";

  const state = {
    stage: "compress",
    query: 15,
    selectedBlock: 2,
    selectedChannel: 2,
    selectedHead: 3,
    selectedGroup: 0,
  };

  const config = {
    n: 16, m: 4, k: 2, window: 2, heads: 8, groups: 2,
    hiddenDim: 8, dim: 6, queryLatentDim: 4, indexDim: 4,
    indexHeads: 4, groupDim: 8,
  };
  const stageMeta = {
    compress: {
      step: "STAGE 1 OF 4",
      title: "Compress KV entries",
      summary: "Every four token positions become one learned, channel-wise summary.",
    },
    index: {
      step: "STAGE 2 OF 4",
      title: "Lightning indexer",
      summary: "A cheap multi-head scorer ranks compressed history before expensive attention runs.",
    },
    mqa: {
      step: "STAGE 3 OF 4",
      title: "Shared key-value multi-query attention",
      summary: "Eight distinct query heads read the same selected memory, with every entry serving as K and V.",
    },
    project: {
      step: "STAGE 4 OF 4",
      title: "Grouped output projection",
      summary: "Head outputs contract inside groups before one final projection returns to model width.",
    },
  };

  const stageWhy = {
    compress: {
      title: "Why compress the KV history?",
      lead: "A dense cache grows one key row and one value row per token. At million-token scale, merely storing and reading that history dominates attention.",
      standardTitle: "Standard dense QKV",
      standardFlow: "H → separate K and V → keep n token rows",
      standardShape: "K, V: n × n<sub>KV</sub> × c",
      deepseekFlow: "H → overlapping learned compressor → joint C<sup>Comp</sup>",
      deepseekShape: "C<sup>Comp</sup>: n/m × c",
      compressed: ["Historical sequence rows: n → n/m", "Separate historical K and V become one joint entry", "Non-local token detail becomes a learned summary"],
      same: ["The compressor still starts from the full hidden states H", "Each output channel is learned independently", "A small local window remains token-level later"],
      reason: "Reduce the dominant long-context KV cache before any query arrives. The a/b overlap lets adjacent compressed entries share boundary evidence instead of cutting context at hard m-token borders.",
      tradeoff: "Compression is lossy: an individual old token can no longer be recovered exactly from C<sup>Comp</sup>. The local window protects recent fine-grained details.",
    },
    index: {
      title: "Why add a lightning indexer?",
      lead: "Compression reduces storage, but core attention over every compressed block would still grow with context length. A cheap retrieval pass decides which blocks deserve full attention.",
      standardTitle: "Standard dense QKV",
      standardFlow: "Every q head scores every cached key",
      standardShape: "attention candidates: n rows",
      deepseekFlow: "narrow indexer scan → top-k block IDs",
      deepseekShape: "K<sup>IComp</sup>: n/m × c<sub>I</sub>; core set: k rows",
      compressed: ["Indexer history is sequence-compressed to n/m", "Indexer width c<sub>I</sub> is narrower than core width c", "Query heads come from low-rank latent c<sup>Q</sup>"],
      same: ["Scores remain content-dependent for the current h<sub>t</sub>", "Multiple heads can search for different signals", "The chosen IDs point back to the real C<sup>Comp</sup> entries"],
      reason: "Spend a small FP4 dot-product scan to avoid a much larger full-width attention scan. The indexer is retrieval; it does not create the values consumed by core attention.",
      tradeoff: "Top-k routing can miss a useful compressed block. It also adds a small independent key cache and scorer, so its scan must remain much cheaper than the work it removes.",
    },
    mqa: {
      title: "Why shared key-value MQA?",
      lead: "Query heads benefit from different read patterns, but duplicating the selected memory for every head would give back much of the cache saving.",
      standardTitle: "Standard multi-head QKV",
      standardFlow: "q<sub>h</sub>, k<sub>h</sub>, v<sub>h</sub> for every head",
      standardShape: "K, V multiply with the KV-head count",
      deepseekFlow: "many q<sub>h</sub> → one shared M<sub>t</sub> used as K = V",
      deepseekShape: "Q: n<sub>h</sub> × c; shared memory: (k+w) × c",
      compressed: ["Long-range memory contains only selected compressed blocks", "One joint entry supplies both key and value", "The memory table is shared across all query heads"],
      same: ["All n<sub>h</sub> query heads remain distinct", "Each head gets its own attention weights and output", "The w local rows remain exact token-level KV"],
      reason: "Preserve head-specific ways of asking questions while storing and reading the expensive context only once. Exact local KV restores short-range precision beside compressed history.",
      tradeoff: "Shared K/V removes per-head memory specialization. DeepSeek keeps specialization on the query side and in each head's attention distribution.",
    },
    project: {
      title: "Why group the output projection?",
      lead: "After MQA, concatenating many wide head outputs creates a large c·n<sub>h</sub> vector. A single direct projection from that width to d becomes expensive.",
      standardTitle: "Standard output projection",
      standardFlow: "concat all heads → one W<sup>O</sup>",
      standardShape: "1 × cn<sub>h</sub> · cn<sub>h</sub> × d",
      deepseekFlow: "split heads → contract each group → concat → final W<sup>O</sup>",
      deepseekShape: "g × (cn<sub>h</sub>/g → d<sub>g</sub>), then gd<sub>g</sub> → d",
      compressed: ["Each group's intermediate width shrinks to d<sub>g</sub>", "The final projection sees gd<sub>g</sub>, not cn<sub>h</sub>", "Projection parameters and multiply-adds are reduced"],
      same: ["Every attention head still contributes", "Group outputs are concatenated before the final mix", "The final output returns to model width d"],
      reason: "Move the bottleneck inside small groups so the final cross-group mixer operates on a compact vector rather than the full head concatenation.",
      tradeoff: "The low-rank group bottleneck constrains how head features interact. The final W<sup>O</sup> restores cross-group mixing after contraction.",
    },
  };

  const makeDense = (rows, columns, fn) => Array.from({ length: rows }, (_, row) =>
    Array.from({ length: columns }, (_, column) => fn(row, column)));
  const makeBias = (fn) => Array.from({ length: config.m }, (_, position) =>
    Array.from({ length: config.dim }, (_, channel) => fn(position, channel)));

  const project = (input, weights) => input.map((row) =>
    Array.from({ length: weights[0].length }, (_, out) =>
      row.reduce((sum, value, inside) => sum + value * weights[inside][out], 0) / Math.sqrt(config.hiddenDim)));

  // H is n×d. Each learned d×c matrix maps hidden width d into compressor width c.
  const H = makeDense(config.n, config.hiddenDim, (t, d) =>
    .83 * Math.sin((t + 1) * .37 + d * .43) + .17 * Math.cos(t * .23 - d * .61));
  const WaKV = makeDense(config.hiddenDim, config.dim, (d, c) => .74 * Math.sin((d + 1) * .41 + c * .53));
  const WbKV = makeDense(config.hiddenDim, config.dim, (d, c) => .70 * Math.cos((d + 2) * .36 - c * .47));
  const WaZ = makeDense(config.hiddenDim, config.dim, (d, c) => .68 * Math.sin((d + 1) * .57 - c * .31));
  const WbZ = makeDense(config.hiddenDim, config.dim, (d, c) => .72 * Math.cos((d + 1) * .49 + c * .38));

  // Four token-level projections from the same H, matching equations (9) and (10).
  const Ca = project(H, WaKV);
  const Cb = project(H, WbKV);
  const Za = project(H, WaZ);
  const Zb = project(H, WbZ);
  const Ba = makeBias((p, c) => .18 * Math.cos((p + 1) * .83 + c * .21));
  const Bb = makeBias((p, c) => .16 * Math.sin((p + 1) * .76 - c * .24));

  function compressionFor(block, channel) {
    const slots = [];
    for (let position = 0; position < config.m; position += 1) {
      const token = block * config.m + position;
      slots.push({ branch: "a", position, token, valid: token < config.n,
        candidate: Ca[token]?.[channel] ?? 0,
        z: Za[token]?.[channel] ?? -Infinity,
        bias: Ba[position][channel] });
    }
    for (let position = 0; position < config.m; position += 1) {
      const token = (block - 1) * config.m + position;
      const valid = block > 0 && token >= 0;
      slots.push({ branch: "b", position, token, valid,
        candidate: valid ? Cb[token][channel] : 0,
        z: valid ? Zb[token][channel] : -Infinity,
        bias: Bb[position][channel] });
    }
    slots.forEach((slot) => { slot.logit = slot.valid ? slot.z + slot.bias : -Infinity; });
    const max = Math.max(...slots.map((slot) => slot.logit));
    const denominator = slots.reduce((sum, slot) => sum + (slot.valid ? Math.exp(slot.logit - max) : 0), 0);
    slots.forEach((slot) => {
      slot.weight = slot.valid ? Math.exp(slot.logit - max) / denominator : 0;
      slot.product = slot.weight * slot.candidate;
    });
    return { slots, value: slots.reduce((sum, slot) => sum + slot.product, 0) };
  }

  const compressed = Array.from({ length: config.n / config.m }, (_, block) =>
    Array.from({ length: config.dim }, (_, channel) => compressionFor(block, channel).value));

  const $ = (id) => document.getElementById(id);
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const fmt = (v) => (Math.round(v * 100) / 100).toFixed(2);

  function shapeNode(symbol, caption, symbolicShape, concreteShape, tone = "violet") {
    const cells = Array.from({ length: 12 }, (_, index) => `<i style="--cell:${(index % 5 + 2) / 7}"></i>`).join("");
    return `<div class="cpipe-shape-node is-${tone}">
      <b>${symbol}</b><small>${caption}</small><em>${symbolicShape} · ${concreteShape}</em>
      <span class="cpipe-shape-glyph" aria-hidden="true">${cells}</span>
    </div>`;
  }

  const traceOp = (symbol, label, wide = false) => `<div class="cpipe-op-node ${wide ? "is-wide" : ""}"><b>${symbol}</b><small>${label}</small></div>`;
  const traceArrow = (label = "") => `<span class="cpipe-trace-arrow" aria-hidden="true">→${label ? `<small>${label}</small>` : ""}</span>`;

  function eligibleBlocks() {
    return Array.from({ length: Math.min(Math.floor(state.query / config.m), compressed.length) }, (_, i) => i);
  }

  function headContribution(block, head) {
    const q = state.query;
    return 0.64 * Math.sin((block + 1) * (head + 2) * 0.73 + q * 0.19)
      + 0.36 * Math.cos((block + 2) * 0.91 - head * 0.38 + q * 0.11);
  }

  function indexScore(block) {
    const gates = [0.62, 0.94, 0.51, 0.77];
    return gates.reduce((sum, gate, head) => sum + gate * Math.max(0, headContribution(block, head)), 0);
  }

  function rankedBlocks() {
    return eligibleBlocks()
      .map((block) => ({ block, score: indexScore(block) }))
      .sort((a, b) => b.score - a.score)
      .map((item, rank) => ({ ...item, rank: rank + 1, selected: rank < config.k }));
  }

  function selectedBlocks() {
    return rankedBlocks().filter((d) => d.selected).map((d) => d.block).sort((a, b) => a - b);
  }

  function memoryEntries() {
    const sparse = selectedBlocks().map((block) => ({ id: `C${block}`, label: `Cᶜᵒᵐᵖ ${block}`, type: "compressed", block }));
    const local = Array.from({ length: config.window }, (_, i) => {
      const token = state.query - config.window + i;
      return { id: `L${token}`, label: `token ${token}`, type: "local", token };
    });
    return [...sparse, ...local];
  }

  function attentionWeights(head) {
    const memory = memoryEntries();
    const logits = memory.map((entry, i) => {
      const base = 0.88 * Math.sin((head + 1) * (i + 1) * 0.61 + state.query * 0.12);
      const locality = entry.type === "local" ? 0.25 + i * 0.07 : 0;
      return base + locality;
    });
    const max = Math.max(...logits);
    const exps = logits.map((v) => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((v) => v / sum);
  }

  function renderHistorySource() {
    const holder = $("cpipe-history-tokens");
    holder.innerHTML = Array.from({ length: config.n }, (_, token) => {
      const isLocal = token >= state.query - config.window && token < state.query;
      const isFuture = token >= state.query;
      return `<i class="${isLocal ? "is-local" : ""} ${isFuture ? "is-future" : ""}"></i>`;
    }).join("");
    $("cpipe-query-sub").textContent = state.query;
  }

  function setStage(stage) {
    state.stage = stage;
    document.querySelectorAll(".cpipe-stage").forEach((button) => {
      const active = button.dataset.stage === stage;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    $("cpipe-flow").dataset.active = stage;
    $("cpipe-query-lane").dataset.active = stage;
    renderDetail();
    renderStageWhy();
  }

  function renderStageWhy() {
    const info = stageWhy[state.stage];
    $("cpipe-stage-drawer-title").textContent = info.title;
    const list = (items) => items.map((item) => `<li>${item}</li>`).join("");
    $("cpipe-stage-drawer-body").innerHTML = `
      <p class="cpipe-drawer-lead">${info.lead}</p>
      <div class="cpipe-schema-compare" aria-label="Standard attention and DeepSeek CSA comparison">
        <section>
          <span>BASELINE</span><h3>${info.standardTitle}</h3>
          <div class="cpipe-schema-flow"><span>${info.standardFlow}</span></div>
          <p>${info.standardShape}</p>
        </section>
        <i aria-hidden="true">vs</i>
        <section class="is-v4">
          <span>DEEPSEEK-V4</span><h3>CSA at this stage</h3>
          <div class="cpipe-schema-flow"><span>${info.deepseekFlow}</span></div>
          <p>${info.deepseekShape}</p>
        </section>
      </div>
      <div class="cpipe-why-answer"><span>WHY DEEPSEEK DID THIS</span><p>${info.reason}</p></div>
      <div class="cpipe-info-compare">
        <section><span class="cpipe-info-mark is-compressed">↓</span><div><h3>Information compressed or shared</h3><ul>${list(info.compressed)}</ul></div></section>
        <section><span class="cpipe-info-mark is-same">=</span><div><h3>Information kept the same or exact</h3><ul>${list(info.same)}</ul></div></section>
      </div>
      <div class="cpipe-tradeoff"><b>What DeepSeek gives up</b><p>${info.tradeoff}</p></div>
      <p class="cpipe-drawer-source"><a href="https://arxiv.org/pdf/2606.19348v1" target="_blank" rel="noreferrer">DeepSeek-V4 §2.3.1 ↗</a></p>`;
  }

  function openDrawer(drawer) {
    if (typeof drawer.showModal === "function") drawer.showModal();
    else drawer.setAttribute("open", "");
  }

  function closeDrawer(drawer) {
    if (typeof drawer.close === "function") drawer.close();
    else drawer.removeAttribute("open");
  }

  function renderDetail() {
    const meta = stageMeta[state.stage];
    $("cpipe-detail-step").textContent = meta.step;
    $("cpipe-detail-title").textContent = meta.title;
    $("cpipe-detail-summary").textContent = meta.summary;
    if (state.stage === "compress") renderCompression();
    if (state.stage === "index") renderIndexer();
    if (state.stage === "mqa") renderMQA();
    if (state.stage === "project") renderProjection();
  }

  function renderCompression() {
    const body = $("cpipe-detail-body");
    const blocks = Array.from({ length: compressed.length }, (_, block) => {
      const tokens = Array.from({ length: config.m }, (_, p) => block * config.m + p);
      const chosen = block === state.selectedBlock;
      return `<button type="button" class="cpipe-compress-block ${chosen ? "is-selected" : ""}" data-block="${block}" aria-pressed="${chosen}">
        <span class="cpipe-block-tokens">${tokens.map((t) => `<i><small>h${t}</small></i>`).join("")}</span>
        <span class="cpipe-down-arrow" aria-hidden="true">↓</span>
        <span class="cpipe-comp-entry"><b>C<sup>Comp</sup><sub>${block}</sub></b><small>1 × c</small></span>
      </button>`;
    }).join("");
    const block = clamp(state.selectedBlock, 0, compressed.length - 1);
    const channel = clamp(state.selectedChannel, 0, config.dim - 1);
    const channelButtons = Array.from({ length: config.dim }, (_, index) => {
      const chosen = index === channel;
      return `<button type="button" data-channel="${index}" class="${chosen ? "is-selected" : ""}" aria-pressed="${chosen}">c${index}</button>`;
    }).join("");

    const currentTokens = Array.from({ length: config.m }, (_, p) => block * config.m + p);
    const previousTokens = Array.from({ length: config.m }, (_, p) => (block - 1) * config.m + p);
    const currentRows = (matrix) => currentTokens.map((token) => matrix[token].slice());
    const previousRows = (matrix) => previousTokens.map((token) => block > 0 ? matrix[token].slice() : Array(matrix[0].length).fill(0));
    const hCurrentRows = currentRows(H);
    const hPreviousRows = previousRows(H);
    const caRows = currentRows(Ca);
    const cbRows = previousRows(Cb);
    const zaRows = currentRows(Za);
    const zbRows = previousRows(Zb);
    const laRows = zaRows.map((row, p) => row.map((value, c) => value + Ba[p][c]));
    const lbRows = zbRows.map((row, p) => row.map((value, c) => block > 0 ? value + Bb[p][c] : -Infinity));
    const allCalcs = Array.from({ length: config.dim }, (_, c) => compressionFor(block, c));
    const saRows = Array.from({ length: config.m }, (_, p) => allCalcs.map((calc) => calc.slots[p].weight));
    const sbRows = Array.from({ length: config.m }, (_, p) => allCalcs.map((calc) => calc.slots[config.m + p].weight));
    const weightedA = caRows.map((row, p) => row.map((value, c) => value * saRows[p][c]));
    const weightedB = cbRows.map((row, p) => row.map((value, c) => value * sbRows[p][c]));

    const matrixNode = (symbol, caption, values, variant, padRows, symbolicShape, highlightChannel = true) => {
      const finite = values.flat().filter(Number.isFinite);
      const max = Math.max(...finite.map((value) => Math.abs(value)), .001);
      const cells = values.map((row, r) => row.map((value, c) => {
        const padded = padRows?.includes(r) || !Number.isFinite(value);
        const classes = ["cpipe-matrix-cell", value < 0 ? "is-negative" : "is-positive",
          variant === "weight" ? "is-weight" : "", padded ? "is-pad" : "", highlightChannel && c === channel ? "is-channel" : ""]
          .filter(Boolean).join(" ");
        const magnitude = Number.isFinite(value) ? Math.abs(value) / max : 0;
        return `<i class="${classes}" style="--mag:${magnitude}"></i>`;
      }).join("")).join("");
      const rows = values.length;
      const columns = values[0]?.length || 0;
      return `<div class="cpipe-matrix-node cpipe-matrix-${variant}" aria-label="${caption}, shape ${rows} by ${columns}">
        <b>${symbol}</b><small>${caption}</small><em>${symbolicShape} · ${rows}×${columns}</em>
        <span class="cpipe-mini-matrix" style="--cols:${columns}">${cells}</span>
      </div>`;
    };

    const op = (symbol, label, wide) => `<div class="cpipe-op-node ${wide ? "is-wide" : ""}"><b>${symbol}</b><small>${label}</small></div>`;
    const pad = block === 0 ? [0, 1, 2, 3] : [];
    const hi = matrixNode("H<sub>i</sub>", "current hidden-state block", hCurrentRows, "hidden", [], "m × d", false);
    const hPrev = matrixNode("H<sub>i−1</sub>", block > 0 ? "previous hidden-state block" : "padded hidden block", hPreviousRows, "hidden", pad, "m × d", false);
    const waKVNode = matrixNode("W<sup>a</sup><sub>KV</sub>", "candidate projection", WaKV, "projection", [], "d × c");
    const wbKVNode = matrixNode("W<sup>b</sup><sub>KV</sub>", "candidate projection", WbKV, "projection", [], "d × c");
    const waZNode = matrixNode("W<sup>a</sup><sub>Z</sub>", "logit projection", WaZ, "projection", [], "d × c");
    const wbZNode = matrixNode("W<sup>b</sup><sub>Z</sub>", "logit projection", WbZ, "projection", [], "d × c");
    const za = matrixNode("Z<sup>a</sup>", "current-block logits", zaRows, "logit", [], "m × c");
    const ba = matrixNode("B<sup>a</sup>", "position bias", Ba, "bias", [], "m × c");
    const la = matrixNode("L<sup>a</sup>", "Zᵃ + Bᵃ", laRows, "logit", [], "m × c");
    const zb = matrixNode("Z<sup>b</sup>", block > 0 ? "previous-block logits" : "padded −∞", zbRows, "logit", pad, "m × c");
    const bb = matrixNode("B<sup>b</sup>", "position bias", Bb, "bias", pad, "m × c");
    const lb = matrixNode("L<sup>b</sup>", block > 0 ? "Zᵇ + Bᵇ" : "remains −∞", lbRows, "logit", pad, "m × c");
    const stackedLogits = matrixNode("[ L<sup>a</sup> ; L<sup>b</sup> ]", "stack 2m rows", [...laRows, ...lbRows], "logit", block === 0 ? [4, 5, 6, 7] : [], "2m × c");
    const stackedWeights = matrixNode("[ S<sup>a</sup> ; S<sup>b</sup> ]", "one shared normalization", [...saRows, ...sbRows], "weight", block === 0 ? [4, 5, 6, 7] : [], "2m × c");
    const sa = matrixNode("S<sup>a</sup>", "current weights", saRows, "weight", [], "m × c");
    const sb = matrixNode("S<sup>b</sup>", "previous weights", sbRows, "weight", pad, "m × c");
    const ca = matrixNode("C<sup>a</sup>", "current KV candidates", caRows, "candidate", [], "m × c");
    const cb = matrixNode("C<sup>b</sup>", block > 0 ? "previous KV candidates" : "zero padding", cbRows, "candidate", pad, "m × c");
    const wa = matrixNode("S<sup>a</sup> ⊙ C<sup>a</sup>", "weighted current candidates", weightedA, "product", [], "m × c");
    const wb = matrixNode("S<sup>b</sup> ⊙ C<sup>b</sup>", "weighted previous candidates", weightedB, "product", pad, "m × c");
    const comp = matrixNode(`C<sup>Comp</sup><sub>${block}</sub>`, "one compressed KV entry", [compressed[block]], "output", [], "1 × c");

    body.innerHTML = `<div class="cpipe-compression-map" role="group" aria-label="Four token blocks compressed into four KV entries">${blocks}</div>
      <div class="cpipe-compression-calc">
        <div class="cpipe-calc-head">
          <div class="cpipe-note-copy">
            <span class="cpipe-eyebrow">BUILDING C<sup>Comp</sup><sub>${block}</sub> · TRACE COLUMN c${channel}</span>
            <b>Follow the matrix operations</b>
            <p>The highlighted column follows one channel. Every other column runs through the same operation tree independently.</p>
          </div>
          <div class="cpipe-channel-picker" role="group" aria-label="Compressed entry channel"><span>trace</span>${channelButtons}</div>
        </div>
        <div class="cpipe-op-tree" role="img" aria-label="Matrix operation tree for one compressed KV entry">
          <div class="cpipe-dimension-rule">
            <span>hidden block</span><b>m × d</b><i>·</i><span>learned projection</span><b>d × c</b><i>→</i><span>compressor features</span><b>m × c</b>
            <small>Toy shapes: ${config.m}×${config.hiddenDim} · ${config.hiddenDim}×${config.dim} → ${config.m}×${config.dim}. The inner d dimension is contracted; c becomes the new feature width.</small>
          </div>
          <div class="cpipe-projection-branches">
            <section class="cpipe-projection-branch cpipe-projection-a">
              <header><b>branch a projections</b><span>same H<sub>i</sub>, two learned maps</span></header>
              <div class="cpipe-projection-fan">
                ${hi}<div class="cpipe-fan-lines"><span>↗</span><span>↘</span></div>
                <div class="cpipe-fan-paths">
                  <div>${op("×", "linear")}${waKVNode}<span class="cpipe-op-arrow">→</span>${ca}</div>
                  <div>${op("×", "linear")}${waZNode}<span class="cpipe-op-arrow">→</span>${za}</div>
                </div>
              </div>
            </section>
            <section class="cpipe-projection-branch cpipe-projection-b">
              <header><b>branch b projections</b><span>${block > 0 ? "same Hᵢ₋₁, separate learned maps" : "previous block is padded"}</span></header>
              <div class="cpipe-projection-fan">
                ${hPrev}<div class="cpipe-fan-lines"><span>↗</span><span>↘</span></div>
                <div class="cpipe-fan-paths">
                  <div>${op("×", "linear")}${wbKVNode}<span class="cpipe-op-arrow">→</span>${cb}</div>
                  <div>${op("×", "linear")}${wbZNode}<span class="cpipe-op-arrow">→</span>${zb}</div>
                </div>
              </div>
            </section>
          </div>
          <div class="cpipe-op-drop"><span>C continues to the multiply · Z forms the softmax weights</span><i>↓</i></div>
          <div class="cpipe-op-branches">
            <section class="cpipe-op-branch cpipe-op-branch-a">
              <header><b>branch a</b><span>current block · tokens ${currentTokens[0]}–${currentTokens[3]}</span></header>
              <div class="cpipe-op-equation">${za}${op("+", "add")}${ba}<span class="cpipe-op-arrow">→</span>${la}</div>
            </section>
            <section class="cpipe-op-branch cpipe-op-branch-b">
              <header><b>branch b</b><span>${block > 0 ? `previous block · tokens ${previousTokens[0]}–${previousTokens[3]}` : "previous block · padded"}</span></header>
              <div class="cpipe-op-equation">${zb}${op("+", "add")}${bb}<span class="cpipe-op-arrow">→</span>${lb}</div>
            </section>
          </div>
          <div class="cpipe-op-drop"><span>stack both branches</span><i>↓</i></div>
          <div class="cpipe-op-center-chain">${stackedLogits}<span class="cpipe-op-arrow">→</span>${op("Softmax", "across all 2m rows", true)}<span class="cpipe-op-arrow">→</span>${stackedWeights}</div>
          <div class="cpipe-op-drop"><span>split weights back into a / b</span><i>↓</i></div>
          <div class="cpipe-op-products">
            <div class="cpipe-op-product cpipe-op-product-a">${sa}${op("⊙", "channel-wise")}${ca}<span class="cpipe-op-arrow">→</span>${wa}</div>
            <div class="cpipe-op-product cpipe-op-product-b">${sb}${op("⊙", "channel-wise")}${cb}<span class="cpipe-op-arrow">→</span>${wb}</div>
          </div>
          <div class="cpipe-op-merge"><span>↘</span>${op("Σ", "sum rows + branches")}<span>↙</span></div>
          <div class="cpipe-op-output">${comp}</div>
        </div>
      </div>`;

    body.querySelectorAll("[data-block]").forEach((button) => button.addEventListener("click", () => {
      state.selectedBlock = Number(button.dataset.block);
      renderCompression();
    }));
    body.querySelectorAll("[data-channel]").forEach((button) => button.addEventListener("click", () => {
      state.selectedChannel = Number(button.dataset.channel);
      renderCompression();
    }));
  }

  function renderIndexer() {
    const body = $("cpipe-detail-body");
    const ranked = rankedBlocks();
    if (!ranked.some((d) => d.block === state.selectedBlock)) state.selectedBlock = ranked[0]?.block ?? 0;
    const maxScore = Math.max(...ranked.map((d) => d.score), 0.01);
    const rows = ranked.map((item) => {
      const chosen = item.block === state.selectedBlock;
      return `<button type="button" class="cpipe-score-row ${item.selected ? "is-kept" : "is-pruned"} ${chosen ? "is-selected" : ""}"
          data-block="${item.block}" aria-pressed="${chosen}">
        <span class="cpipe-score-id"><b>C<sup>Comp</sup><sub>${item.block}</sub></b><small>rank #${item.rank}</small></span>
        <span class="cpipe-score-track"><i style="--score:${item.score / maxScore}"></i></span>
        <span class="cpipe-score-value">${fmt(item.score)}</span>
        <span class="cpipe-score-decision">${item.selected ? "KEEP" : "PRUNE"}</span>
      </button>`;
    }).join("");
    const chosen = ranked.find((d) => d.block === state.selectedBlock) || ranked[0];
    const gates = [0.62, 0.94, 0.51, 0.77];
    const heads = gates.map((gate, head) => {
      const raw = headContribution(chosen.block, head);
      const relu = Math.max(0, raw);
      return `<div class="cpipe-index-head">
        <span>head ${head + 1}</span><b>${fmt(gate)} × ReLU(${fmt(raw)})</b><em>${fmt(gate * relu)}</em>
      </div>`;
    }).join("");

    const hiddenSequence = shapeNode("H", "same layer hidden states", "n × d", `${config.n}×${config.hiddenDim}`, "gold");
    const indexKeys = shapeNode("K<sup>IComp</sup>", "compressed indexer keys", "n/m × c<sub>I</sub>", `${config.n / config.m}×${config.indexDim}`);
    const queryHidden = shapeNode(`h<sub>${state.query}</sub>`, "current query hidden state", "1 × d", `1×${config.hiddenDim}`, "green");
    const downWeight = shapeNode("W<sup>DQ</sup>", "query down-projection", "d × d<sub>c</sub>", `${config.hiddenDim}×${config.queryLatentDim}`, "neutral");
    const queryLatent = shapeNode("c<sup>Q</sup><sub>t</sub>", "shared query latent", "1 × d<sub>c</sub>", `1×${config.queryLatentDim}`, "green");
    const indexUpWeight = shapeNode("W<sup>IUQ</sup>", "indexer up-projection", "d<sub>c</sub> × c<sub>I</sub>n<sup>I</sup><sub>h</sub>", `${config.queryLatentDim}×${config.indexDim * config.indexHeads}`, "neutral");
    const indexQueries = shapeNode("q<sup>I</sup><sub>t</sub>", "reshape into indexer heads", "n<sup>I</sup><sub>h</sub> × c<sub>I</sub>", `${config.indexHeads}×${config.indexDim}`, "green");
    const gateWeight = shapeNode("W<sup>w</sup>", "head-weight projection", "d × n<sup>I</sup><sub>h</sub>", `${config.hiddenDim}×${config.indexHeads}`, "neutral");
    const headWeights = shapeNode("w<sup>I</sup><sub>t</sub>", "one scalar per head", "1 × n<sup>I</sup><sub>h</sub>", `1×${config.indexHeads}`, "orange");
    const selectedKey = shapeNode(`K<sup>IComp</sup><sub>${chosen.block}</sub>`, `indexer key for block ${chosen.block}`, "1 × c<sub>I</sub>", `1×${config.indexDim}`);
    const headDots = shapeNode("q<sup>I</sup><sub>t</sub> K<sup>IComp</sup><sub>s</sub>", "one dot product per head", "n<sup>I</sup><sub>h</sub> × 1", `${config.indexHeads}×1`, "green");
    const score = shapeNode(`I<sub>${state.query},${chosen.block}</sub>`, "one ranking score", "1 × 1", "1×1", "orange");

    body.innerHTML = `<div class="cpipe-trace-intro">
      <span class="cpipe-eyebrow">WHERE THE INDEXER INPUTS COME FROM</span>
      <b>History builds keys; the current token builds queries and head weights.</b>
      <p>K<sup>IComp</sup> is <strong>not projected from C<sup>Comp</sup></strong>. It starts from the same H but uses its own indexer-specific compressor and its own narrower width c<sub>I</sub>.</p>
    </div>
    <div class="cpipe-origin-grid cpipe-index-origins">
      <section class="cpipe-trace-section">
        <header><b>HISTORY PATH · built once for KV blocks</b><span>same compression recipe, separate learned parameters</span></header>
        <div class="cpipe-trace-chain">${hiddenSequence}${traceArrow()}${traceOp("2m→1", "indexer compressor", true)}${traceArrow()}${indexKeys}</div>
        <p class="cpipe-trace-caption">Internally this repeats stage 1's candidate + logit + positional-bias + shared-softmax operation, but outputs c<sub>I</sub>-wide index keys.</p>
      </section>
      <section class="cpipe-trace-section">
        <header><b>QUERY PATH · recomputed for token t</b><span>low-rank latent fans into indexer heads</span></header>
        <div class="cpipe-trace-chain">${queryHidden}${traceOp("×", "linear")}${downWeight}${traceArrow()}${queryLatent}</div>
        <div class="cpipe-trace-fan">
          <span class="cpipe-trace-fan-source">c<sup>Q</sup><sub>t</sub></span>
          <span aria-hidden="true">↘</span>
          <div class="cpipe-trace-chain is-compact">${indexUpWeight}${traceArrow("reshape")}${indexQueries}</div>
        </div>
        <div class="cpipe-trace-chain is-compact cpipe-weight-path">${queryHidden}${traceOp("×", "linear")}${gateWeight}${traceArrow()}${headWeights}</div>
      </section>
    </div>
    <div class="cpipe-score-operation" aria-label="Indexer score operation for compressed block ${chosen.block}">
      <div class="cpipe-note-copy">
        <span class="cpipe-eyebrow">SCORE BLOCK ${chosen.block}</span>
        <b>Four head comparisons collapse to one scalar</b>
        <p>Dot each query head with the block's shared index key, apply ReLU, multiply by its learned head weight, then sum.</p>
      </div>
      <div class="cpipe-trace-chain">${indexQueries}${traceOp("·", "head-wise dot")}${selectedKey}${traceArrow()}${headDots}${traceOp("ReLU", "clip negatives", true)}${traceOp("⊙", "multiply wᴵ")}${traceOp("Σ", "sum heads")}${traceArrow()}${score}</div>
    </div>
    <div class="cpipe-index-results">
      <div class="cpipe-detail-note cpipe-index-note">
        <div class="cpipe-note-copy">
          <span class="cpipe-eyebrow">INDEX SCORE FOR C<sup>Comp</sup><sub>${chosen.block}</sub></span>
          <b>Weighted evidence across indexer heads</b>
          <p>Each head contributes w<sup>I</sup><sub>h</sub> · ReLU(q<sup>I</sup><sub>h</sub> · K<sup>IComp</sup>). Contributions sum to ${fmt(chosen.score)}.</p>
        </div>
        <div class="cpipe-index-heads">${heads}</div>
      </div>
      <div class="cpipe-score-list" aria-label="Ranked index scores">${rows}</div>
    </div>`;

    body.querySelectorAll("[data-block]").forEach((button) => button.addEventListener("click", () => {
      state.selectedBlock = Number(button.dataset.block);
      renderIndexer();
    }));
  }

  function renderMQA() {
    const body = $("cpipe-detail-body");
    const memory = memoryEntries();
    const heads = Array.from({ length: config.heads }, (_, head) => {
      const chosen = head === state.selectedHead;
      const weights = attentionWeights(head);
      return `<button type="button" class="cpipe-attn-row ${chosen ? "is-selected" : ""}" data-head="${head}" aria-pressed="${chosen}">
        <span class="cpipe-q-head">q<sub>${head + 1}</sub></span>
        <span class="cpipe-attn-cells">${weights.map((weight, i) => `<i class="${memory[i].type}" style="--alpha:${weight}" aria-label="${memory[i].label}: ${Math.round(weight * 100)} percent"><small>${Math.round(weight * 100)}%</small></i>`).join("")}</span>
        <span class="cpipe-head-output">o<sub>${head + 1}</sub></span>
      </button>`;
    }).join("");
    const chosenWeights = attentionWeights(state.selectedHead);
    const strongestIndex = chosenWeights.indexOf(Math.max(...chosenWeights));
    const strongest = memory[strongestIndex];

    const queryLatent = shapeNode("c<sup>Q</sup><sub>t</sub>", "reused from stage 2", "1 × d<sub>c</sub>", `1×${config.queryLatentDim}`, "green");
    const queryUpWeight = shapeNode("W<sup>UQ</sup>", "attention-query up-projection", "d<sub>c</sub> × cn<sub>h</sub>", `${config.queryLatentDim}×${config.dim * config.heads}`, "neutral");
    const allQueries = shapeNode("q<sub>t</sub>", "reshape into attention heads", "n<sub>h</sub> × c", `${config.heads}×${config.dim}`, "green");
    const sparseMemory = shapeNode("C<sup>SprsComp</sup><sub>t</sub>", "top-k entries from stage 2", "k × c", `${config.k}×${config.dim}`);
    const localHidden = shapeNode("H<sup>local</sup><sub>t</sub>", "exact sliding-window states", "w × d", `${config.window}×${config.hiddenDim}`, "gold");
    const localWeight = shapeNode("W<sup>KV</sup>", "token-level KV projection", "d × c", `${config.hiddenDim}×${config.dim}`, "neutral");
    const localMemory = shapeNode("C<sup>local</sup><sub>t</sub>", "uncompressed local KV", "w × c", `${config.window}×${config.dim}`, "orange");
    const sharedMemory = shapeNode("M<sub>t</sub> = K = V", "one shared memory table", "(k+w) × c", `${config.k + config.window}×${config.dim}`);
    const oneQuery = shapeNode(`q<sub>t,${state.selectedHead + 1}</sub>`, `query head ${state.selectedHead + 1}`, "1 × c", `1×${config.dim}`, "green");
    const oneOutput = shapeNode(`o<sub>t,${state.selectedHead + 1}</sub>`, "one head output", "1 × c", `1×${config.dim}`, "orange");

    body.innerHTML = `<div class="cpipe-trace-intro">
      <span class="cpipe-eyebrow">TWO INPUT PATHS MEET IN CORE ATTENTION</span>
      <b>The query latent expands into many Q heads; memory stays shared.</b>
      <p>The same c<sup>Q</sup><sub>t</sub> created for the lightning indexer is reused here. Sparse compressed history and exact local tokens are concatenated once, then serve as both K and V for every head.</p>
    </div>
    <div class="cpipe-origin-grid cpipe-mqa-origins">
      <section class="cpipe-trace-section">
        <header><b>QUERY HEADS</b><span>stage 2 latent → eight c-wide queries</span></header>
        <div class="cpipe-trace-chain">${queryLatent}${traceOp("×", "linear")}${queryUpWeight}${traceArrow("reshape")}${allQueries}</div>
      </section>
      <section class="cpipe-trace-section">
        <header><b>SHARED MEMORY</b><span>selected long-range blocks + exact local detail</span></header>
        <div class="cpipe-memory-merge">
          <div class="cpipe-trace-chain is-compact">${sparseMemory}<span class="cpipe-memory-source-tag">from stages 1–2</span></div>
          <span class="cpipe-merge-mark" aria-hidden="true">+</span>
          <div class="cpipe-trace-chain is-compact">${localHidden}${traceOp("×", "linear")}${localWeight}${traceArrow()}${localMemory}</div>
        </div>
        <div class="cpipe-trace-drop"><span>concatenate rows</span><i>↓</i></div>
        <div class="cpipe-trace-output">${sharedMemory}</div>
      </section>
    </div>
    <div class="cpipe-core-operation">
      <div class="cpipe-note-copy">
        <span class="cpipe-eyebrow">ONE HEAD · SAME OPERATION REPEATS ${config.heads}×</span>
        <b>Query changes by head; K and V do not.</b>
        <p>Head ${state.selectedHead + 1} scores the shared rows with qKᵀ, softmaxes those scores, and uses them to sum the same rows as values.</p>
      </div>
      <div class="cpipe-trace-chain">${oneQuery}${traceOp("Attn", "qKᵀ → softmax → ·V", true)}${sharedMemory}${traceArrow()}${oneOutput}</div>
    </div>
    <div class="cpipe-mqa-layout">
      <div class="cpipe-memory-head">
        <span class="cpipe-memory-label">shared K = V</span>
        <div class="cpipe-memory-entries">${memory.map((entry) => `<span class="${entry.type}"><b>${entry.label}</b><small>${entry.type === "local" ? "exact local KV" : "selected compressed KV"}</small></span>`).join("")}</div>
      </div>
      <div class="cpipe-attn-grid" role="group" aria-label="Eight query heads attending to one shared KV set">
        <div class="cpipe-attn-axis"><span>query</span><span>attention over shared memory</span><span>head output</span></div>
        ${heads}
      </div>
    </div>
    <div class="cpipe-selection-line"><b>Head ${state.selectedHead + 1}</b> places its largest weight on <strong>${strongest.label}</strong> (${Math.round(chosenWeights[strongestIndex] * 100)}%). The KV rows are stored once—not repeated for eight heads.</div>`;

    body.querySelectorAll("[data-head]").forEach((button) => button.addEventListener("click", () => {
      state.selectedHead = Number(button.dataset.head);
      renderMQA();
    }));
  }

  function renderProjection() {
    const body = $("cpipe-detail-body");
    const perGroup = config.heads / config.groups;
    const groups = Array.from({ length: config.groups }, (_, group) => {
      const chosen = group === state.selectedGroup;
      const start = group * perGroup;
      const heads = Array.from({ length: perGroup }, (_, i) => `<span>o<sub>${start + i + 1}</sub></span>`).join("");
      return `<button type="button" class="cpipe-proj-group ${chosen ? "is-selected" : ""}" data-group="${group}" aria-pressed="${chosen}">
        <span class="cpipe-group-label">GROUP ${group + 1}</span>
        <span class="cpipe-group-heads">${heads}</span>
        <span class="cpipe-proj-arrow">1×${perGroup * config.dim} ↓ W<sup>G</sup><sub>${group + 1}</sub> · ${perGroup * config.dim}×${config.groupDim}</span>
        <span class="cpipe-group-latent"><b>o′<sup>G</sup><sub>${group + 1}</sub></b><small>1×d<sub>g</sub> · 1×${config.groupDim}</small></span>
      </button>`;
    }).join("");

    const allHeadOutputs = shapeNode("o<sub>t</sub>", "all MQA head outputs", "1 × cn<sub>h</sub>", `1×${config.dim * config.heads}`, "orange");
    const groupInput = shapeNode(`o<sup>G</sup><sub>t,${state.selectedGroup + 1}</sub>`, `${perGroup} heads in selected group`, "1 × cn<sub>h</sub>/g", `1×${perGroup * config.dim}`, "orange");
    const groupWeight = shapeNode(`W<sup>G</sup><sub>${state.selectedGroup + 1}</sub>`, "group contraction", "cn<sub>h</sub>/g × d<sub>g</sub>", `${perGroup * config.dim}×${config.groupDim}`, "neutral");
    const groupOutput = shapeNode(`o′<sup>G</sup><sub>t,${state.selectedGroup + 1}</sub>`, "compact group output", "1 × d<sub>g</sub>", `1×${config.groupDim}`);
    const compactAll = shapeNode("[o′<sup>G</sup><sub>t,1</sub>; …; o′<sup>G</sup><sub>t,g</sub>]", "concatenated group outputs", "1 × d<sub>g</sub>g", `1×${config.groupDim * config.groups}`);
    const finalWeight = shapeNode("W<sup>O</sup>", "final output projection", "d<sub>g</sub>g × d", `${config.groupDim * config.groups}×${config.hiddenDim}`, "neutral");
    const finalOutput = shapeNode("ô<sub>t</sub>", "attention output at model width", "1 × d", `1×${config.hiddenDim}`, "green");

    body.innerHTML = `<div class="cpipe-trace-intro">
      <span class="cpipe-eyebrow">WHY THE OUTPUT PROJECTION HAS TWO LEVELS</span>
      <b>Contract small head groups first; mix the compact results once.</b>
      <p>Directly mapping all ${config.heads} × ${config.dim} head features to d would use one large c n<sub>h</sub> × d projection. Grouping inserts a narrower d<sub>g</sub> bottleneck before the final mix.</p>
    </div>
    <div class="cpipe-projection-trace">
      <div class="cpipe-trace-chain cpipe-projection-entry">${allHeadOutputs}${traceArrow("split into g groups")}${groupInput}</div>
      <div class="cpipe-group-operation">
        <div class="cpipe-note-copy">
          <span class="cpipe-eyebrow">TRACE GROUP ${state.selectedGroup + 1}</span>
          <b>${perGroup} head outputs contract from ${perGroup * config.dim} to ${config.groupDim} features</b>
          <p>Every group has the same input/output shapes but its own learned slice of the grouped projection.</p>
        </div>
        <div class="cpipe-trace-chain">${groupInput}${traceOp("×", "linear")}${groupWeight}${traceArrow()}${groupOutput}</div>
      </div>
    </div>
    <div class="cpipe-projection-flow" aria-label="Grouped output projection">
      <div class="cpipe-wide-output"><span>core attention output</span><b>[ o<sub>1</sub> · o<sub>2</sub> · o<sub>3</sub> · o<sub>4</sub> · o<sub>5</sub> · o<sub>6</sub> · o<sub>7</sub> · o<sub>8</sub> ]</b><small>1 × c n<sub>h</sub> · 1×${config.dim * config.heads}</small></div>
      <span class="cpipe-proj-down">split by heads ↓</span>
      <div class="cpipe-proj-groups">${groups}</div>
      <span class="cpipe-proj-down">concatenate compact group outputs ↓</span>
      <div class="cpipe-final-chain">${compactAll}${traceOp("×", "linear")}${finalWeight}${traceArrow()}${finalOutput}</div>
    </div>
    <div class="cpipe-selection-line"><b>Group ${state.selectedGroup + 1}</b> maps 1×${perGroup * config.dim} → 1×${config.groupDim}. Concatenating both groups gives 1×${config.groupDim * config.groups}, which W<sup>O</sup> returns to model width 1×${config.hiddenDim}.</div>`;

    body.querySelectorAll("[data-group]").forEach((button) => button.addEventListener("click", () => {
      state.selectedGroup = Number(button.dataset.group);
      renderProjection();
    }));
  }

  document.querySelectorAll(".cpipe-stage").forEach((button) => button.addEventListener("click", () => setStage(button.dataset.stage)));
  $("cpipe-stage-why").addEventListener("click", () => {
    renderStageWhy();
    openDrawer($("cpipe-stage-drawer"));
  });
  $("cpipe-efficiency-open").addEventListener("click", () => openDrawer($("cpipe-efficiency-drawer")));
  document.querySelectorAll("[data-close-drawer]").forEach((button) => button.addEventListener("click", () => closeDrawer(button.closest("dialog"))));
  document.querySelectorAll(".cpipe-drawer").forEach((drawer) => drawer.addEventListener("click", (event) => {
    if (event.target === drawer) closeDrawer(drawer);
  }));

  renderHistorySource();
  setStage("compress");
}());
