# =============================================================================
# WHITE NOISE SAMPLING ON GRAPHS
# functions.py
#
# All functions for the white noise sampling on graphs project.
# Based on: Vassilevski (2026), "White Noise Sampling on Graphs"
#
# Usage:
#   from functions import *
#
# For live reloading in JupyterLab:
#   %load_ext autoreload
#   %autoreload 2
#   from functions import *
#
# Authors: Diya Patel, Prof. Panayot S. Vassilevski
# Portland State University, Summer 2026
# =============================================================================

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import eigh, cho_factor, cho_solve
from scipy.io import mmread
from tqdm import tqdm
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# =============================================================================
# DATA LOADING
# =============================================================================

def load_graph(filepath):
    """
    Load a graph from either a Matrix Market (.mtx) file or a plain edge list.
    Automatically detects format and whether edges are weighted.

    Parameters
    ----------
    filepath : str — path to the graph file

    Returns
    -------
    edges        : list of (i, j) tuples — 0-indexed
    n_vertices   : int — total number of vertices
    edge_weights : dict of {(i,j): weight} or None if unweighted
    """
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()

    if first_line.startswith('%%MatrixMarket'):
        sparse_matrix = mmread(filepath).tocsr()
        n_vertices = sparse_matrix.shape[0]
        coo = sparse_matrix.tocoo()
        edges = []
        edge_weights = {}
        for i, j, v in zip(coo.row, coo.col, coo.data):
            if i < j:
                edges.append((int(i), int(j)))
                if v != 1.0:
                    edge_weights[(int(i), int(j))] = float(v)
        edge_weights = edge_weights if edge_weights else None

    else:
        edges = []
        edge_weights = {}
        max_vertex = 0
        weighted = None

        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('%') or line.startswith('#'):
                    continue
                parts = line.split()
                if weighted is None:
                    weighted = len(parts) == 3
                i, j = int(parts[0]) - 1, int(parts[1]) - 1
                if i == j:
                    continue
                i, j = min(i, j), max(i, j)
                edges.append((i, j))
                if weighted:
                    edge_weights[(i, j)] = float(parts[2])
                max_vertex = max(max_vertex, i, j)

        edges = list(set(edges))
        n_vertices = max_vertex + 1
        edge_weights = edge_weights if weighted else None

    print(f"✓ Loaded: {filepath}")
    print(f"  Vertices : {n_vertices}")
    print(f"  Edges    : {len(edges)}")
    print(f"  Weighted : {'yes' if edge_weights else 'no'}")
    print()
    return edges, n_vertices, edge_weights


# =============================================================================
# PHASE 1: Graph Setup
# =============================================================================

def build_graph_matrices(edges, n_vertices, edge_weights=None):
    """
    Build adjacency matrix A, degree matrix D, and graph Laplacian L = D - A.

    Parameters
    ----------
    edges        : list of (i, j) tuples
    n_vertices   : int
    edge_weights : dict of {(i,j): weight} or None for unweighted

    Returns
    -------
    A : np.ndarray (n x n) — adjacency matrix
    D : np.ndarray (n x n) — diagonal degree matrix
    L : np.ndarray (n x n) — graph Laplacian
    """
    A = np.zeros((n_vertices, n_vertices))

    for (i, j) in edges:
        w = edge_weights.get((i,j), edge_weights.get((j,i), 1.0)) if edge_weights else 1.0
        A[i, j] = w
        A[j, i] = w

    degrees = A.sum(axis=1)
    D = np.diag(degrees)
    L = D - A

    # Sanity checks
    assert np.allclose(A, A.T), "A is not symmetric!"
    assert np.allclose(L, L.T), "L is not symmetric!"
    assert abs(np.linalg.eigvalsh(L)[0]) < 1e-8, "L does not have a zero eigenvalue!"

    print("✓ Phase 1 complete: A, D, L built and verified")
    print(f"  Matrix size : {n_vertices} x {n_vertices}")
    print(f"  Degree range: [{int(degrees.min())}, {int(degrees.max())}]")
    print()
    return A, D, L


# =============================================================================
# PHASE 2: Spectral Shift
# =============================================================================

def compute_lambda_min(L, D):
    """
    Compute the smallest positive eigenvalue of D^{-1}L by solving
    the generalized eigenvalue problem Lv = lambda Dv.

    Parameters
    ----------
    L : np.ndarray (n x n) — graph Laplacian
    D : np.ndarray (n x n) — degree matrix

    Returns
    -------
    lambda_min : float — smallest positive eigenvalue of D^{-1}L
    """
    eigenvalues, _ = eigh(L, D)
    positive = eigenvalues[eigenvalues > 1e-10]
    assert len(positive) > 0, "No positive eigenvalues found!"
    lambda_min = positive[0]
    print(f"✓ lambda_min = {lambda_min:.6f}")
    return lambda_min


def build_shifted_laplacian(L, D, lambda_min, sigma_squared=0.25):
    """
    Build the shifted Laplacian L_sigma = L + sigma^2 * lambda_min * D.
    L_sigma is symmetric positive definite (SPD) and invertible.

    Parameters
    ----------
    L             : np.ndarray — graph Laplacian
    D             : np.ndarray — degree matrix
    lambda_min    : float — from compute_lambda_min
    sigma_squared : float — shift parameter (default 0.25)

    Returns
    -------
    L_sigma : np.ndarray — shifted Laplacian (SPD)
    """
    L_sigma = L + sigma_squared * lambda_min * D

    # Sanity check
    assert np.linalg.eigvalsh(L_sigma)[0] > 0, "L_sigma is not SPD!"

    print(f"✓ Phase 2 complete: L_sigma built (sigma^2 = {sigma_squared})")
    print(f"  Smallest eigenvalue of L_sigma: {np.linalg.eigvalsh(L_sigma)[0]:.6f}")
    print()
    return L_sigma

# sanity check phase 2
def sanity_check_phase2(L_sigma):
    """
    Run sanity checks on L_sigma from Phase 2.
    """
    print("=== Phase 2 Sanity Checks ===")
 
    # Check symmetry
    assert np.allclose(L_sigma, L_sigma.T), "L_sigma is not symmetric!"
    print("✓ L_sigma is symmetric")
 
    # Check all eigenvalues are positive (SPD condition)
    eigenvalues = np.linalg.eigvalsh(L_sigma)
    min_eig = eigenvalues[0]
    print(f"✓ Smallest eigenvalue of L_sigma: {min_eig:.6f} (should be > 0)")
    assert min_eig > 0, "L_sigma is not SPD — has non-positive eigenvalue!"
    print("✓ L_sigma is SPD (all eigenvalues positive)")
    print()

# =============================================================================
# PHASE 3: White Noise Sampling Pipeline
# =============================================================================

def build_incidence_matrix(edges, n_vertices):
    """
    Build the signed vertex-edge incidence matrix B (n x m).
    B[i, e] = +1 if vertex i is the first endpoint of edge e
    B[j, e] = -1 if vertex j is the second endpoint of edge e

    Parameters
    ----------
    edges      : list of (i, j) tuples
    n_vertices : int

    Returns
    -------
    B : np.ndarray (n_vertices x n_edges)
    """
    B = np.zeros((n_vertices, len(edges)))
    for idx, (i, j) in enumerate(edges):
        B[i, idx] =  1
        B[j, idx] = -1
    return B

def draw_edge_noise(n_edges):
    """
    STEP 3a: Draw independent N(0,1) noise for every edge.

    Parameters
    ----------
    n_edges : int

    Returns
    -------
    w : np.ndarray of shape (n_edges,)
    """
    w = np.random.randn(n_edges)
    return w


def map_noise_to_vertices(B, w):
    """
    STEP 3b: Map edge noise to vertices.
    f_i = sum of w_e over all edges touching vertex i.
    In matrix form: f = B @ w

    Parameters
    ----------
    B : np.ndarray (n_vertices x n_edges)
    w : np.ndarray (n_edges,)

    Returns
    -------
    f : np.ndarray (n_vertices,)
    """
    f = B @ w
    return f


def solve_for_u(L_sigma, lambda_min, f, cho_cache=None):
    """
    STEP 3c: Solve L_sigma u = lambda_min * f.
    Cholesky factorization is computed once and reused every MC step
    since L_sigma is fixed across all samples.

    Parameters
    ----------
    L_sigma    : np.ndarray (n x n) — fixed across all MC steps
    lambda_min : float
    f          : np.ndarray (n,)
    cho_cache  : precomputed Cholesky factorization (reuse across steps)

    Returns
    -------
    u         : np.ndarray (n,)
    cho_cache : pass this back in on the next MC step
    """
    if cho_cache is None:
        cho_cache = cho_factor(L_sigma)

    rhs = lambda_min * f
    u = cho_solve(cho_cache, rhs)

    return u, cho_cache

def run_phase3(L_sigma, lambda_min, edges, n_vertices):
    """
    Run the full Phase 3 white noise sampling pipeline.

    Steps:
        3a. Draw edge noise:       w_e ~ N(0,1) for every edge
        3b. Map noise to vertices: f = B @ w
        3c. Solve for u:           L_sigma @ u = lambda_min * f

    L_sigma is Cholesky factorized once for reuse in the MC loop.

    Parameters
    ----------
    L_sigma    : np.ndarray — shifted Laplacian from Phase 2
    lambda_min : float — from Phase 2
    edges      : list of (i, j) tuples
    n_vertices : int

    Returns
    -------
    u         : np.ndarray (n_vertices,) — spatially correlated field
    B         : np.ndarray — incidence matrix (reuse in MC loop)
    cho_cache : Cholesky factorization (reuse in MC loop)
    """
    B         = build_incidence_matrix(edges, n_vertices)
    cho_cache = cho_factor(L_sigma)

    # 3a
    w = np.random.randn(len(edges))
    # 3b
    f = B @ w
    # 3c
    u = cho_solve(cho_cache, lambda_min * f)

    print("✓ Phase 3 complete: correlated field u generated")
    print(f"  u range: [{u.min():.4f}, {u.max():.4f}]")
    print()
    return u, B, cho_cache


# =============================================================================
# PHASE 4: Build Random Edge Permeability
# =============================================================================

def compute_permeability(u, edges, edge_weights=None):
    """
    Convert vertex field u into edge permeabilities k.
    For each edge e = (i,j): k_e = exp((u_i + u_j) / 2)

    Exponentiation ensures k_e > 0 always (log-normal distribution).
    If original edge weights exist, they are multiplied into k_e.

    Parameters
    ----------
    u            : np.ndarray (n_vertices,) — from Phase 3
    edges        : list of (i, j) tuples
    edge_weights : dict or None — from load_graph

    Returns
    -------
    k : dict of {(i,j): k_e}
    """
    k = {}
    for (i, j) in edges:
        base = np.exp((u[i] + u[j]) / 2)
        if edge_weights:
            w = edge_weights.get((i,j), edge_weights.get((j,i), 1.0))
            k[(i, j)] = base * w
        else:
            k[(i, j)] = base

    assert all(v > 0 for v in k.values()), "Negative permeability detected!"

    print("✓ Phase 4 complete: permeabilities computed")
    print(f"  min k_e : {min(k.values()):.4f}")
    print(f"  max k_e : {max(k.values()):.4f}")
    print(f"  mean k_e: {np.mean(list(k.values())):.4f}")
    print()
    return k


# =============================================================================
# PHASE 5: Weighted Laplacian Solve (Darcy Flow)
# =============================================================================

def build_weighted_laplacian(edges, n_vertices, k):
    """
    Build the weighted graph Laplacian L_k = sum_e k_e * d_e * d_e^T
    using sparse matrix format for efficiency.

    Same structure as L from Phase 1 but each edge contributes
    proportionally to its permeability k_e rather than uniformly.
    High permeability edges dominate, low permeability edges barely
    contribute. Stored in CSR format for fast arithmetic and solving.

    Parameters
    ----------
    edges      : list of (i, j) tuples
    n_vertices : int
    k          : dict of {(i,j): k_e} — permeabilities from Phase 4

    Returns
    -------
    L_k : scipy.sparse.csr_matrix (n x n)
    """
    L_k = lil_matrix((n_vertices, n_vertices))
    for (i, j) in edges:
        k_e = k.get((i,j), k.get((j,i), 1.0))
        L_k[i, i] += k_e
        L_k[j, j] += k_e
        L_k[i, j] -= k_e
        L_k[j, i] -= k_e
    return L_k.tocsr()


def solve_darcy(L_k, n_vertices, gamma_in, gamma_out, p_in=1.0, p_out=0.0):
    """
    Solve Darcy flow using efficient sparse matrix slicing.
    """
    p = np.zeros(n_vertices)
    for v in gamma_in:
        p[v] = p_in
    for v in gamma_out:
        p[v] = p_out

    boundary = set(gamma_in) | set(gamma_out)
    interior = np.array([v for v in range(n_vertices) if v not in boundary])

    # efficient sparse slicing — avoid np.ix_ on sparse matrices
    L_interior = L_k[interior, :][:, interior]
    rhs        = -L_k[interior, :][:, list(boundary)] @ p[list(boundary)]

    p_interior = spsolve(L_interior, rhs)
    p[interior] = p_interior

    return p

# =============================================================================
# PHASE 6: Quantity of Interest
# =============================================================================

def extract_qoi(p, k, edges, gamma_out):
    """
    Compute the quantity of interest Q — total flux out of gamma_out.
    Q = sum of k_e * (p_i - p_j) for edges incident to gamma_out.

    Parameters
    ----------
    p         : np.ndarray (n_vertices,) — from Phase 5
    k         : dict of {(i,j): k_e} — from Phase 4
    edges     : list of (i, j) tuples
    gamma_out : list of int — outlet vertex indices

    Returns
    -------
    Q : float — total outlet flux
    """
    gamma_out_set = set(gamma_out)
    Q = 0.0
    for (i, j) in edges:
        k_e = k.get((i,j), k.get((j,i), 1.0))
        if i in gamma_out_set or j in gamma_out_set:
            Q += k_e * abs(p[i] - p[j])
    return Q


# =============================================================================
# PHASE 7: Monte Carlo Loop
# =============================================================================

def monte_carlo_loop(L_sigma, lambda_min, edges, n_vertices,
                     gamma_in, gamma_out, edge_weights=None, N=1000):
    """
    Full Monte Carlo loop — runs Phases 3-6 N times and records Q each step.
    L_sigma is Cholesky factorized once and reused every step.

    Parameters
    ----------
    L_sigma      : np.ndarray — shifted Laplacian from Phase 2
    lambda_min   : float — from Phase 2
    edges        : list of (i, j) tuples
    n_vertices   : int
    gamma_in     : list of int — inlet vertices
    gamma_out    : list of int — outlet vertices
    edge_weights : dict or None
    N            : int — number of Monte Carlo samples

    Returns
    -------
    Q_samples : np.ndarray (N,) — one Q value per sample
    """
    B         = build_incidence_matrix(edges, n_vertices)
    cho_cache = cho_factor(L_sigma)
    Q_samples = np.zeros(N)

    for i in range(N):
        w  = np.random.randn(len(edges))
        f  = B @ w
        u  = cho_solve(cho_cache, lambda_min * f)
        k  = {(vi,vj): np.exp((u[vi]+u[vj])/2) * (edge_weights.get((vi,vj), edge_weights.get((vj,vi), 1.0)) if edge_weights else 1.0)
              for (vi,vj) in edges}
        L_k = build_weighted_laplacian(edges, n_vertices, k)
        p   = solve_darcy(L_k, n_vertices, gamma_in, gamma_out)
        Q_samples[i] = extract_qoi(p, k, edges, gamma_out)

        if (i+1) % 100 == 0:
            print(f"  Step {i+1}/{N} | running mean Q: {Q_samples[:i+1].mean():.4f}")

    return Q_samples


def analyze_qoi(Q_samples, save_as=None):
    """
    Analyze and plot the distribution of Q across all Monte Carlo samples.
    Produces a histogram and a convergence plot of the running mean.

    Parameters
    ----------
    Q_samples : np.ndarray (N,) — from monte_carlo_loop
    """
    print("\n=== Monte Carlo Results ===")
    print(f"  Samples : {len(Q_samples)}")
    print(f"  Mean Q  : {Q_samples.mean():.6f}")
    print(f"  Std Q   : {Q_samples.std():.6f}")
    print(f"  Var Q   : {Q_samples.var():.6f}")
    print(f"  Min Q   : {Q_samples.min():.6f}")
    print(f"  Max Q   : {Q_samples.max():.6f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f1117")

    for ax in axes:
        ax.set_facecolor("#0f1117")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")

    # Histogram
    axes[0].hist(Q_samples, bins=40, color="#5bc8f5", edgecolor="#0f1117", alpha=0.9)
    axes[0].axvline(Q_samples.mean(), color="red", linestyle="--",
                    linewidth=2, label=f"Mean = {Q_samples.mean():.4f}")
    axes[0].set_xlabel("Q (total flux)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of QoI")
    axes[0].legend(facecolor="#0f1117", labelcolor="white")

    # Convergence
    running_mean = np.cumsum(Q_samples) / np.arange(1, len(Q_samples)+1)
    axes[1].plot(running_mean, color="#5bc8f5", linewidth=1.5)
    axes[1].axhline(Q_samples.mean(), color="red", linestyle="--",
                    linewidth=2, label=f"Final mean = {Q_samples.mean():.4f}")
    axes[1].set_xlabel("Number of samples")
    axes[1].set_ylabel("Running mean of Q")
    axes[1].set_title("Monte Carlo Convergence")
    axes[1].legend(facecolor="#0f1117", labelcolor="white")

    plt.tight_layout()
    # save the image before displaying it
    if save_as:
        plt.savefig(save_as, dpi=150, bbox_inches="tight")
        print(f"Saved as {save_as}")
    plt.show()

def monte_carlo_loop_tqdm(L_sigma, lambda_min, edges, n_vertices,
                          gamma_in, gamma_out, edge_weights=None, N=1000):
    """
    Monte Carlo loop with tqdm progress bar.
    Runs Phases 3-6 N times and records Q each step.
    """
    B         = build_incidence_matrix(edges, n_vertices)
    cho_cache = cho_factor(L_sigma)
    Q_samples = np.zeros(N)

    with tqdm(total=N, desc="Monte Carlo", unit="sample") as pbar:
        for i in range(N):
            # Phase 3
            w = np.random.randn(len(edges))
            f = B @ w
            u = cho_solve(cho_cache, lambda_min * f)

            # Phase 4
            k = {(vi, vj): np.exp((u[vi] + u[vj]) / 2) *
                 (edge_weights.get((vi,vj), edge_weights.get((vj,vi), 1.0))
                  if edge_weights else 1.0)
                 for (vi, vj) in edges}

            # Phase 5
            L_k = build_weighted_laplacian(edges, n_vertices, k)
            p   = solve_darcy(L_k, n_vertices, gamma_in, gamma_out)

            # Phase 6
            Q_samples[i] = extract_qoi(p, k, edges, gamma_out)

            pbar.update(1)
            pbar.set_postfix({"mean Q": f"{Q_samples[:i+1].mean():.4f}"})

    print(f"\n✓ Done — {N} samples complete")
    print(f"  Mean Q : {Q_samples.mean():.6f}")
    print(f"  Std Q  : {Q_samples.std():.6f}")
    print(f"  Var Q  : {Q_samples.var():.6f}")

    return Q_samples


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_graph_2d(edges, n_vertices, edge_weights=None, title="Graph",
                       save_as=None):
    """
    Pretty 2D graph visualization with degree-based node sizing and coloring.

    Parameters
    ----------
    edges        : list of (i, j) tuples
    n_vertices   : int
    edge_weights : dict or None
    title        : str
    save_as      : str or None — filename to save the figure (e.g. 'graph.png')
    """
    G = nx.Graph()
    G.add_nodes_from(range(n_vertices))

    if edge_weights:
        for (i, j) in edges:
            w = edge_weights.get((i,j), edge_weights.get((j,i), 1.0))
            G.add_edge(i, j, weight=w)
    else:
        G.add_edges_from(edges)

    degrees     = dict(G.degree())
    node_sizes  = [300 + degrees[v] * 100 for v in G.nodes()]
    node_colors = [degrees[v] for v in G.nodes()]
    pos         = nx.spring_layout(G, seed=42, k=2)

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    if edge_weights:
        weights = [G[i][j]['weight'] for i,j in G.edges()]
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=weights,
                               edge_cmap=plt.cm.plasma, width=1.5,
                               alpha=0.6, edge_vmin=min(weights),
                               edge_vmax=max(weights))
        sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma,
                                   norm=plt.Normalize(vmin=min(weights),
                                   vmax=max(weights)))
        plt.colorbar(sm, ax=ax, label="Edge Weight", shrink=0.5)
    else:
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="white",
                               width=0.8, alpha=0.3)

    nc = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                                node_color=node_colors, cmap=plt.cm.cool,
                                alpha=0.95)
    cb = plt.colorbar(nc, ax=ax, shrink=0.5)
    cb.set_label("Degree", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    if n_vertices <= 100:
        nx.draw_networkx_labels(G, pos, ax=ax, font_color="black", font_size=7)

    ax.set_title(title, color="white", fontsize=16, pad=20)
    ax.axis("off")
    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi=150, bbox_inches="tight")
        print(f"Saved as {save_as}")

    plt.show()


def visualize_graph_3d(edges, n_vertices, edge_weights=None, title="Graph (3D)",
                       save_as=None):
    """
    3D graph visualization with degree-based node sizing and coloring.

    Parameters
    ----------
    edges        : list of (i, j) tuples
    n_vertices   : int
    edge_weights : dict or None
    title        : str
    save_as      : str or None — filename to save the figure
    """
    G = nx.Graph()
    G.add_nodes_from(range(n_vertices))
    G.add_edges_from(edges)

    degrees = dict(G.degree())
    pos_2d  = nx.spring_layout(G, seed=42, k=2)
    np.random.seed(42)
    pos_3d  = {v: np.array([pos_2d[v][0], pos_2d[v][1],
                             np.random.uniform(-1, 1)]) for v in G.nodes()}

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("#0f1117")
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_facecolor("#0f1117")

    for (i, j) in G.edges():
        x = [pos_3d[i][0], pos_3d[j][0]]
        y = [pos_3d[i][1], pos_3d[j][1]]
        z = [pos_3d[i][2], pos_3d[j][2]]
        ax.plot(x, y, z, color="white", alpha=0.15, linewidth=0.4)

    xs          = [pos_3d[v][0] for v in G.nodes()]
    ys          = [pos_3d[v][1] for v in G.nodes()]
    zs          = [pos_3d[v][2] for v in G.nodes()]
    node_colors = [degrees[v] for v in G.nodes()]
    node_sizes  = [10 + degrees[v] * 8 for v in G.nodes()]

    sc = ax.scatter(xs, ys, zs, c=node_colors, cmap=plt.cm.plasma,
                    s=node_sizes, alpha=0.9, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="Degree", shrink=0.5, pad=0.1)

    ax.set_title(title, color="white", fontsize=16, pad=20)
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#0f1117')
    ax.yaxis.pane.set_edgecolor('#0f1117')
    ax.zaxis.pane.set_edgecolor('#0f1117')
    ax.tick_params(colors='white')
    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi=150, bbox_inches="tight")
        print(f"Saved as {save_as}")

    plt.show()


# visualisation functins using plotly for interactive graphs
import plotly.graph_objects as go

def visualize_graph_2d_plotly(edges, n_vertices, edge_weights=None, title="Graph"):
    """
    Interactive 2D graph visualization using Plotly.
    Hover over nodes to see vertex index and degree.
    """
    G = nx.Graph()
    G.add_nodes_from(range(n_vertices))
    if edge_weights:
        for (i, j) in edges:
            w = edge_weights.get((i,j), edge_weights.get((j,i), 1.0))
            G.add_edge(i, j, weight=w)
    else:
        G.add_edges_from(edges)

    degrees = dict(G.degree())
    pos     = nx.spring_layout(G, seed=42, k=2)

    # Edge traces
    edge_x, edge_y = [], []
    for (i, j) in G.edges():
        x0, y0 = pos[i]
        x1, y1 = pos[j]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=0.8, color='rgba(255,255,255,0.3)'),
        hoverinfo='none'
    )

    # Node traces
    node_x      = [pos[v][0] for v in G.nodes()]
    node_y      = [pos[v][1] for v in G.nodes()]
    node_degrees = [degrees[v] for v in G.nodes()]
    node_text   = [f"Vertex {v}<br>Degree: {degrees[v]}" for v in G.nodes()]
    node_sizes  = [10 + degrees[v] * 5 for v in G.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            size=node_sizes,
            color=node_degrees,
            colorscale='Plasma',
            showscale=True,
            colorbar=dict(title="Degree", thickness=15),
            line=dict(width=0.5, color='white')
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text=title, font=dict(color='white', size=16)),
            paper_bgcolor='#0f1117',
            plot_bgcolor='#0f1117',
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=20, r=20, t=50, b=20)
        )
    )
    fig.show()


def visualize_graph_3d_plotly(edges, n_vertices, edge_weights=None, title="Graph (3D)"):
    """
    Interactive 3D graph visualization using Plotly.
    Fully rotatable — drag to explore the graph structure.
    Node size and color reflect degree.
    """
    G = nx.Graph()
    G.add_nodes_from(range(n_vertices))
    G.add_edges_from(edges)

    degrees = dict(G.degree())

    # 3D spring layout using networkx 2D + spectral for z
    pos_2d = nx.spring_layout(G, seed=42, k=2)
    
    # use second eigenvector of laplacian for z coordinate (more meaningful than random)
    L_mat   = nx.laplacian_matrix(G).toarray().astype(float)
    eigvals, eigvecs = np.linalg.eigh(L_mat)
    z_coord = eigvecs[:, 1]  # Fiedler vector — captures graph structure in z

    pos_3d = {v: (pos_2d[v][0], pos_2d[v][1], z_coord[v]) for v in G.nodes()}

    # edge traces
    edge_x, edge_y, edge_z = [], [], []
    for (i, j) in G.edges():
        x0, y0, z0 = pos_3d[i]
        x1, y1, z1 = pos_3d[j]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(width=1, color='rgba(255,255,255,0.3)'),
        hoverinfo='none',
        showlegend=False
    )

    # node traces
    node_x       = [pos_3d[v][0] for v in G.nodes()]
    node_y       = [pos_3d[v][1] for v in G.nodes()]
    node_z       = [pos_3d[v][2] for v in G.nodes()]
    node_degrees = np.array([degrees[v] for v in G.nodes()])
    node_sizes = [3 + degrees[v] * 0.5 for v in G.nodes()]
    node_text    = [f"Vertex {v}<br>Degree: {degrees[v]}" for v in G.nodes()]

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers+text' if n_vertices <= 50 else 'markers',
        hoverinfo='text',
        text=[str(v) for v in G.nodes()] if n_vertices <= 50 else None,
        hovertext=node_text,
        marker=dict(
            size=node_sizes,
            color=node_degrees,
            colorscale='Viridis',
            cmin=min(node_degrees),
            cmax=max(node_degrees),
            showscale=True,
            colorbar=dict(
                title=dict(text="Degree", font=dict(color="white")),
                tickfont=dict(color="white"),
                thickness=15
            ),
            line=dict(width=0.5, color='white'),
            opacity=0.9
        ),
    showlegend=False
    )

    plotly_fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text=title, font=dict(color='white', size=16)),
            paper_bgcolor='#0f1117',
            height=800,
            width=1200,
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False,
                           showticklabels=False, backgroundcolor='#0f1117',
                           showspikes=False),
                yaxis=dict(showgrid=False, zeroline=False,
                           showticklabels=False, backgroundcolor='#0f1117',
                           showspikes=False),
                zaxis=dict(showgrid=False, zeroline=False,
                           showticklabels=False, backgroundcolor='#0f1117',
                           showspikes=False),
                bgcolor='#0f1117',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            margin=dict(l=0, r=0, t=50, b=0)
        )
    )
    plotly_fig.show()
    return plotly_fig
