# White Noise Sampling on Graphs

A Monte Carlo simulation framework for sampling spatially correlated random fields on graphs, implemented as part of a DOE/NSF-funded summer research fellowship at Portland State University.

**Authors:** Diya Patel, Prof. Panayot S. Vassilevski  
**Institution:** Fariborz Maseeh Department of Mathematics and Statistics, Portland State University  
**Funding:** U.S. Department of Energy, Office of Science, Advanced Scientific Computing Research Program; NSF RTG grant DMS-2136228

---

## Overview

This project extends by analogy a known Monte Carlo sampling procedure for discretized PDEs (partial differential equations) to graph Laplacian problems. The framework models uncertain flow through networks — such as neural signal propagation in brain networks or fluid flow through porous media — by sampling spatially correlated random permeability fields on graph edges and solving the resulting Darcy flow problem.

The primary dataset used is the **Rhesus Macaque Brain Connectivity Network** (90 brain regions, structural neural connections), where the quantity of interest Q measures total signal flux from peripheral sensory regions to central processing hubs under uncertain synaptic connection strengths.

---

## Project Structure

```
summer-project/
│
├── functions.py          # All function definitions (load, build, solve, visualize)
├── demo.ipynb            # Full pipeline demo on Les Misérables network
├── rhesus_brain.ipynb    # Full pipeline on Rhesus Macaque Brain Network
├── power_grid.ipynb      # Full pipeline on US Power Grid
├── data_cleaning.ipynb   # Data loading and preprocessing
│
├── data/
│   ├── rhesus_brain.mtx  # Rhesus macaque brain connectivity network
│   ├── power-US-Grid.mtx # US Power Grid network
│   └── football.mtx      # NCAA football network (demo)
│
└── figures/
    ├── rhesus_phase1.png  # Graph visualization
    ├── rhesus_phase2.png  # Eigenvalue spectrum
    ├── rhesus_phase3.png  # White noise sampling (f vs u)
    ├── rhesus_phase4.png  # Edge permeability field
    ├── rhesus_phase5_6.png # Pressure field
    └── rhesus_phase7.png  # Monte Carlo results
```

---

## Pipeline

The implementation follows a 7-phase Monte Carlo pipeline:

**Phase 1 — Graph Setup**  
Load graph data and construct the adjacency matrix A, degree matrix D, and graph Laplacian L = D - A.

**Phase 2 — Spectral Shift**  
Compute λ⁺_min, the smallest positive eigenvalue of D⁻¹L, and form the shifted Laplacian Lσ = L + σ²λ⁺_min D (σ² = 0.25). Lσ is symmetric positive definite (SPD) and invertible.

**Phase 3 — White Noise Sampling**  
Generate a spatially correlated random field u on the vertices via three steps:
- Draw independent N(0,1) noise w_e for every edge
- Map to vertices: f = B @ w (B = signed incidence matrix)
- Solve: $$\lambda^+_{\min} f$$ (Cholesky factorized once, reused every MC step)

**Phase 4 — Edge Permeability**  
Convert vertex field u to edge permeabilities: $$k_e = \exp\left(\frac{u_i + u_j}{2}\right)$$. Ensures k_e > 0 always (log-normal distribution).

**Phase 5 — Darcy Flow Solve**  
Build weighted Laplacian $$L_k = \sum_e k_e \mathbf{d}_e \mathbf{d}_e^T$$ and solve the flow problem with boundary conditions p = 1 on Γ_in and p = 0 on Γ_out.

**Phase 6 — Quantity of Interest**  
Extract $$Q = \sum_{e=(i,j) \in \Gamma_{out}} k_e (p_i - p_j)$$ over outlet edges — total flux exiting through Γ_out.

**Phase 7 — Monte Carlo Loop**  
Repeat Phases 3-6 N = 1000 times. Analyze the distribution of Q: mean, variance, and convergence.

---

## Installation

```bash
pip install numpy scipy matplotlib networkx plotly tqdm pyamg
```

---

## Usage

```python
# import all functions
from functions import *

# load graph
edges, n_vertices, edge_weights = load_graph("rhesus_brain.mtx")

# Phase 1
A, D, L = build_graph_matrices(edges, n_vertices, edge_weights)

# Phase 2
lambda_min = compute_lambda_min(L, D)
L_sigma    = build_shifted_laplacian(L, D, lambda_min)

# Phase 3
import numpy as np
np.random.seed(42)
u, B, cho_cache = run_phase3(L_sigma, lambda_min, edges, n_vertices)

# Phase 4
k = compute_permeability(u, edges, edge_weights)

# Phase 5 & 6
degrees_arr = A.sum(axis=1)
gamma_in    = list(np.argsort(degrees_arr)[:5])   # peripheral regions
gamma_out   = list(np.argsort(degrees_arr)[-5:])  # hub regions

L_k = build_weighted_laplacian(edges, n_vertices, k)
p   = solve_darcy(L_k, n_vertices, gamma_in, gamma_out)
Q   = extract_qoi(p, k, edges, gamma_out)

# Phase 7
from tqdm import tqdm
Q_samples = monte_carlo_loop_tqdm(L_sigma, lambda_min, edges, n_vertices,
                                   gamma_in, gamma_out, N=1000)
analyze_qoi(Q_samples, save_as="results.png")
```

---

## Dataset

**Rhesus Macaque Brain Connectivity Network**
- 90 vertices — brain regions parcellated using a standard brain atlas
- Edges — structural white matter tract connections between regions
- Undirected and unweighted
- Source: Network Data Repository (networkrepository.com)

**Γ_in:** 20 lowest degree peripheral regions (sensory input)  
**Γ_out:** 5 highest degree hub regions (central processing)  
**Q:** Total neural signal flux from peripheral to hub regions under uncertain synaptic connection strengths

---

## Key Design Decisions

- **Cholesky prefactorization:** Lσ is fixed across all MC steps so it is factorized once using `scipy.linalg.cho_factor` and reused every step — significant speedup for large N
- **Log-normal permeability:** k_e = exp(...) ensures positivity and produces a realistic distribution consistent with porous media and biological systems
- **Degree-based Γ_in/Γ_out:** Hub nodes serve as outlets (signal sinks) and peripheral nodes as inlets (signal sources), consistent with known neuroscience — sensory signals originate at peripheral regions and propagate inward to processing hubs

---

## Extensions (Future Work)

- Multilevel Monte Carlo (MLMC) — solve on hierarchy of coarser graphs for variance reduction
- Markov Chain Monte Carlo (MCMC) — replace independent samples with a Markov chain
- Sparse solvers (PCG + AMG) — scale to larger graphs beyond dense solver limits
- Application to US Power Grid and road networks

---

## References

- Vassilevski, P.S. (2026). *White Noise Sampling on Graphs*. Portland State University.
- Rossi, R.A. and Ahmed, N.K. (2015). *The Network Data Repository with Interactive Graph Analytics and Visualization*. AAAI.
