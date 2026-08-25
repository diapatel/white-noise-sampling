# Code Walkthrough — Line-by-Line Rationale

Going through `functions_v2.py` phase by phase: what each piece of code does, why it's written that way, and any issues found along the way.

---

## Phase 1 — Graph Setup (A, D, L)

### Building the adjacency matrix `A`

```python
A = lil_matrix((n_vertices, n_vertices))
for (i, j) in edges:
    w = edge_weights.get((i,j), edge_weights.get((j,i), 1.0)) if edge_weights else 1.0
    A[i, j] = w
    A[j, i] = w
A = A.tocsr()
```

**What it's doing:** builds the adjacency matrix by inserting one (or two, for undirected symmetry) entries per edge, then converts the result to a different sparse format for downstream use.

**Why `lil_matrix` first, not `csr_matrix` directly:**
- **LIL (List of Lists)** stores each row as a list of `(column, value)` pairs — designed to be **cheap to build incrementally**, one entry at a time, via normal indexing like `A[i, j] = w`.
- **CSR (Compressed Sparse Row)** stores the whole matrix as three flat arrays (`data`, `indices`, `indptr`) — designed to be **fast for math operations** (matrix-vector products, row slicing) but **expensive to modify incrementally**, since inserting a new entry requires shifting/reallocating the underlying flat arrays.

If you tried to build a matrix directly in CSR format via a loop of individual insertions, it would still work, but be slow — scipy will even emit a `SparseEfficiencyWarning` if you try.

**The idiom:** use LIL for the "assembly" phase (looping over edges), then call `.tocsr()` once at the end to convert for everything downstream (Phase 3's `B @ w`, Phase 5b's interior/boundary slicing, etc., all benefit from CSR's fast math). Pay the one-time conversion cost once, rather than paying LIL's slower math performance repeatedly throughout the rest of the pipeline.

**The loop body:**
- `A[i, j] = w` **and** `A[j, i] = w` — sets both symmetric entries per edge, since the graph is undirected (edge `(i,j)` means mutual connection, so both directions of the adjacency matrix get the same weight).
- `edge_weights.get((i,j), edge_weights.get((j,i), 1.0)) if edge_weights else 1.0` — looks up the edge weight: tries `(i,j)` first, falls back to `(j,i)` in case the dict stored it in the other order, falls back to `1.0` (unweighted) if `edge_weights` is `None` or the edge isn't found. Ties back to this project's convention of discarding native dataset weights (`edge_weights=None`) for most datasets — every edge then gets weight `1.0`, i.e. plain unweighted adjacency.

**Verdict:** correct and efficient as written. The LIL→CSR pattern is the standard, deliberate choice here — not something to change.

---

*(Continue below as we walk through the rest of Phase 1, then Phases 2-7)*
