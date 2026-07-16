# Collective Communication, visually

An interactive, dependency-free explainer for the collective communication
operations behind distributed deep-learning training. The goal is something you
can *play with* to build intuition — tune the number of GPUs, scrub through the
algorithm step by step, inspect the actual data each GPU holds, and see the
data-volume / time cost model.

Implemented operations (switch via the Operation dropdown): **All-Gather**,
**Reduce-Scatter**, **All-Reduce** (ring), plus **Scatter**, **Broadcast**, and
**All-to-All**. They all share one collective-agnostic renderer — each algorithm
is just a pure state machine in `src/model.js`.

A second, standalone page — `gather.html` (linked from the masthead nav) —
visualizes the **tensor-indexing** ops `torch.gather` and `torch.scatter`: how
the `index` redirects exactly one axis (`dim`) while the other axis stays pinned
to its own position. Scrub through the mapping one element at a time and watch
the arrow stay inside a row (`dim=1`) or a column (`dim=0`). The page closes with
a worked **cross-entropy loss** example — gathering each position's gold-token
log-prob from the `[T, V]` `log_softmax` output — plus where gather/scatter show
up elsewhere (MoE routing, beam-search KV-cache reorder, one-hot labels).

## Features

- **Tunable ring size** — 2–8 GPUs.
- **Step scrubber + play/pause** — watch chunks propagate one neighbor per round.
- **Tappable submatrices** — each chunk is a real `a × b` weight submatrix
  (Glorot-style values in `[-1, 1]`); tap any block to inspect its numbers. In
  reduce-scatter, tapping shows the **running partial sum** and which GPUs'
  contributions have been reduced in so far (a `k/N` badge per slot).
- **Tunable chunk shape** — adjust rows `a` and cols `b` live.
- **Full-tensor view** — assemble all chunks into the complete global tensor.
- **Cost model panel** (collapsible, at the bottom) — live data volume and time:
  - shard `S = a·b·dtype`, full tensor `D = N·S`
  - per-GPU bytes sent `(N−1)·S = (N−1)/N · D`
  - wall-clock time `T = (N−1)·S / W` for per-device bandwidth `W`
  - derivation of why ring all-gather is **bandwidth-optimal**.

## Run it

It's plain HTML + SVG + JavaScript with **no build step**.

- Easiest: open `index.html` directly in a browser.
- Or serve the folder (avoids any future module/CORS issues):

  ```sh
  python -m http.server 5500
  # then visit http://localhost:5500
  ```

## Project structure

| File            | Role |
|-----------------|------|
| `index.html`    | Layout: controls, stage, inspector, cost panel |
| `gather.html`   | Standalone page for `torch.gather` / `torch.scatter` |
| `src/gather.js` | Step machine + renderer for the indexing-ops page |
| `styles.css`    | Styling + pop/fly animations |
| `src/data.js`   | Global tensor → per-chunk submatrices, weights, colors |
| `src/model.js`  | Collective algorithms as pure step-by-step state machines |
| `src/render.js` | Node cards, tappable chunk cells, arrows, flying blocks |
| `src/main.js`   | Wires controls, inspector, and the cost model |

To add a new collective, write a builder in `src/model.js` that returns the same
`{ name, numNodes, steps[] }` shape — the renderer needs no changes.

## Visualization conventions

Apply these rules when adding or revising technical visualizations in this
repository:

- **Always label matrix shapes.** Put a shape hint directly beside every matrix
  or tensor node. Prefer both the symbolic shape (for example, `m × d`) and the
  concrete teaching-example shape (for example, `4 × 8`) when space permits.
- **Design for desktop.** Prioritize the clarity and spatial organization of the
  desktop visualization. Mobile-specific visualization layouts and responsive
  optimization are not required.
- **Keep technical text comfortably readable.** Use at least `12px` for normal
  labels and explanatory text. Reserve `11px` only for secondary annotations,
  such as matrix-shape hints; important labels and values should generally be
  `13–15px` or larger. Never shrink essential text merely to fit more content.

## Roadmap

- [x] All-Gather
- [x] Reduce-Scatter
- [x] All-Reduce (reduce-scatter + all-gather; cost `2·(N−1)/N·D`)
- [x] Scatter
- [x] Broadcast
- [x] All-to-All (bidirectional-ring / TPU torus, ⌊N/2⌋ steps)
- [ ] Optional latency (α) term in the cost model
- [ ] Tree / pipelined variants for broadcast & scatter
