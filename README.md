# White Noise Sampling on Graphs

Extends the PDE-based white noise sampling technique of Osborn, Vassilevski, and Villa (2017) to graph Laplacian settings. Builds a full Monte Carlo pipeline for estimating flow-related quantities on graphs, and a two-level Monte Carlo variance-reduction method with an empirically validated precondition (**boundary fraction**) for when it succeeds.

Summer research fellowship project, Portland State University (DOE/NSF funded). Advisor: Prof. Panayot Vassilevski.

## What this does

Given a graph, the pipeline:
1. Builds the graph Laplacian and a shifted, invertible version of it
2. Draws random white noise and solves for a spatially correlated random field on the graph's vertices
3. Converts that field into random, spatially correlated edge permeabilities
4. Solves a Darcy-flow-style boundary value problem (fixed source/sink pressure) on the resulting weighted graph
5. Extracts a scalar quantity of interest (Q) — total flux through the sink
6. Repeats via Monte Carlo to estimate E[Q]

A two-level Monte Carlo extension coarsens the graph (vertex aggregation) and combines many cheap coarse-graph samples with a small number of expensive paired fine/coarse samples, reducing the number of full-resolution solves needed for an accurate estimate.

## Repository structure

```
src/
  functions_v2.py      # Phases 0-7: load data, build matrices, Monte Carlo loop
  two_level_mc.py       # aggregation, coarse graph construction, paired sampling
  graph_mlmc_model.py   # adapter implementing a collaborator's MultilevelModel protocol
data/
  raw/                  # datasets (edge lists / Matrix Market files)
notebooks/
  pipeline_runs/         # per-dataset pipeline runs
  dev_and_scratch/       # development / exploratory notebooks
outputs/
  figures/               # generated plots and figures
```

## Pipeline phases

| Phase | What it does |
|---|---|
| 0 | Load graph data (`.mtx` or edge-list), remap to contiguous vertex IDs, check/filter connectivity |
| 1 | Build adjacency (A), degree (D), and Laplacian (L = D - A) matrices, sparse |
| 2 | Find λ_min (smallest positive eigenvalue), build shifted Laplacian L_σ = L + σ²λ_min D (SPD, invertible) |
| 3 | Draw white noise w, map to vertices (f = Bw), solve L_σu = √λ_min·f for the spatially correlated field u (sparse Cholesky) |
| 4 | Compute edge permeability k_e = exp((u_i + u_j)/2) |
| 5 | Build the weighted Laplacian L_k, solve the Darcy flow boundary value problem for pressure p |
| 6 | Extract Q = Σ k_e\|p_i - p_j\| over edges touching the sink boundary |
| 7 | Repeat Phases 3-6 across N samples for a Monte Carlo estimate of E[Q] |

### Gamma selection (choosing source/sink boundary vertices)

Three strategies, matched to a dataset's structure:
- **Degree-based** — hub vertices as source, low-degree periphery as sink (hub-and-spoke graphs)
- **Diameter-endpoint** — double-BFS to approximate the graph's diameter, expand each endpoint into a k-hop neighborhood (spatially-extended graphs)
- **Community detection** — greedy modularity maximization, two largest communities as source/sink (graphs with strong modular structure)

## Two-level Monte Carlo

```
E[Q] ≈ E[Q_coarse] (many cheap samples) + E[Q_fine - Q_coarse] (few expensive paired samples)
```

Fine and coarse permeability are derived from the same random draw each sample, keeping the correction term low-variance.

### Key finding: boundary fraction

```
boundary fraction = (|Γ_in| + |Γ_out|) / n
```

Aggregation can only coarsen the interior portion of the graph. Established empirically across six datasets:
- **Under ~1%** — reliably succeeds
- **~25%+** — reliably fails (insufficient interior vertices remain after coarsening)

For diameter-endpoint selection, boundary fraction depends on k_hop **relative to the graph's diameter**, not k_hop alone — the same k_hop can succeed on one graph and fail catastrophically on another with a shorter diameter.

### Key finding: structural validity is necessary but not sufficient

Passing the boundary-fraction and interior-vertex checks does not guarantee success. For community-detection-based gamma selection specifically, **modularity of the detected communities** is a strong additional predictor — low modularity (weakly separated communities) can produce poor correlation between Q_fine and Q_coarse even when every structural precondition passes.

## Results summary

| Dataset | n | Gamma method | Boundary fraction | Correlation | Variance reduction |
|---|---|---|---|---|---|
| US Power Grid | 4,941 | Diameter-endpoint | 0.75% | 1.0000 | 3.11x |
| Oregon Router | 11,174 | Diameter-endpoint | 0.92% | 0.9995 | 31.25x |
| bio-grid-yeast | 6,008 | Diameter-endpoint (k_hop=1) | 0.07% | 1.0000 | Strong |
| Facebook Ego Network | 4,039 | Community (top-5% subset) | 2.20% | 0.9956 | 12.66x |
| Rhesus Brain | 90 | Degree-based | ~25% | — | Failed (zero interior coarse vertices) |
| bio-CE-GN | 2,215 | Community (whole) | ~70% | — | Not run (predicted failure) |
| C. elegans | 2,215 | Community (top-5% subset) | 3.43% | 0.72 | 1.05x (no benefit despite passing structural checks) |
| Minnesota Roads | — | Diameter-endpoint | — | — | Ran, but unresolved convergence instability |

## Dependencies

```
numpy, scipy, networkx, scikit-sparse (CHOLMOD), tqdm, matplotlib
```

Sparse Cholesky factorization (via `scikit-sparse`) gives approximately a 500x speedup over dense Cholesky on large graphs, validated on Oregon Router; the shifted Laplacian's factorization is computed once and reused across every Monte Carlo sample.

## MLMC framework integration

`graph_mlmc_model.py` adapts this pipeline to a collaborator's generic multi-level Monte Carlo framework (`MultilevelModel` protocol: `mlmc_model.py`, `mlmc_correction.py`, `mlmc_statistics.py`, `linear_solver.py`, `mlmc_runner.py`), so it can be run through their `MLMCRunner` alongside a PDE-based implementation of the same protocol — enabling direct comparison between the two approaches on shared statistics and reporting infrastructure.

Validated to match this repo's own independent results (`two_level_mc.py`, run directly) on US Power Grid and Oregon Router, both in estimate value and runtime. In the course of validating this integration, a bug was found and fixed in the adapter's handling of the framework's level-0 convention (the coarsest level's "fine" slot must represent the coarse quantity, not the fine one) — see `graph_mlmc_model.py`'s docstring for details.

The collaborator's framework does not currently support cached sparse Cholesky factorization; the adapter routes the Phase 3 white-noise solve around their generic solver, calling this repo's own cached factorization directly, to preserve the ~500x speedup described above. Only the Darcy flow solve is routed through their solver.

## Status

Active summer research fellowship project. Results validated across six datasets, plus cross-validated against a collaborator's independent multi-level Monte Carlo framework on two of them. Not yet published.
