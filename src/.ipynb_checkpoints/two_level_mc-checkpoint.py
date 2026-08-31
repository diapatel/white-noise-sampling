"""
two_level_mc.py

Two-Level Monte Carlo extension for the white-noise-sampling-on-graphs
pipeline. Builds a coarse graph via vertex aggregation, then runs paired
fine/coarse Monte Carlo samples using correlated randomness to estimate
E[Q_fine] more cheaply than direct fine-only sampling.

Depends on functions.py for: build_incidence_matrix, sparse_cholesky
(from sksparse.cholmod), and the standard Phase 1-2 pipeline outputs
(edges, n_vertices, L_sigma, lambda_min).

--------------------------------------------------------------------
VALIDATED ONLY WHEN THE BOUNDARY FRACTION IS SMALL. Always run
check_boundary_fraction() before attempting aggregation on a new
dataset -- see that function's docstring for what "small" means and
why this matters. This code has failed outright on datasets where
gamma_in/gamma_out claim a large share of the graph (see project notes
for the Rhesus Brain and bio-CE-GN case studies).
--------------------------------------------------------------------
"""

import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

# ============================================================
# Precondition check -- run this BEFORE attempting aggregation
# ============================================================

def check_boundary_fraction(gamma_in, gamma_out, n_vertices, verbose=True):
    """
    Compute what fraction of the graph is claimed by gamma_in + gamma_out.

    This is the single most important diagnostic for whether two-level
    Monte Carlo is likely to work on a given dataset. Aggregation can
    only coarsen the non-boundary portion of the graph -- if boundary
    vertices dominate, the coarse graph can end up with zero interior
    vertices, making the Darcy solve degenerate (no unknowns to solve
    for).

    Rules of thumb established across six tested datasets:
        - Under ~1%   : aggregation reliably succeeds
                         (US Power Grid: 0.75%, Oregon Router: 0.92%,
                          bio-grid-yeast with corrected k_hop: 0.07%)
        - ~25% or more : aggregation fails outright
                         (Rhesus Brain: 25% -> zero interior coarse
                          vertices; bio-CE-GN: 70%, bio-grid-yeast: 76%
                          via community detection -- both predicted/
                          confirmed to fail for the same reason)

    For diameter-endpoint gamma selection specifically, boundary
    fraction depends heavily on k_hop RELATIVE TO THE GRAPH'S DIAMETER,
    not on k_hop alone. A fixed k_hop that works on a large-diameter
    graph (e.g. k_hop=4 on a diameter-46 power grid) can catastrophically
    fail on a small-diameter, hub-dominated graph (e.g. the same k_hop=4
    claimed 99.87% of bio-grid-yeast, whose diameter is only 5 hops).
    Always check the graph's diameter and scale k_hop accordingly
    (e.g. k_hop = max(1, diameter // 10)) rather than using a fixed
    constant across datasets.

    Parameters
    ----------
    gamma_in, gamma_out : list of int
    n_vertices          : int
    verbose             : bool -- print a pass/fail-style summary

    Returns
    -------
    boundary_fraction : float
    """
    boundary_fraction = (len(gamma_in) + len(gamma_out)) / n_vertices
    if verbose:
        print(f"gamma_in: {len(gamma_in)}, gamma_out: {len(gamma_out)}, "
              f"n_vertices: {n_vertices}")
        print(f"Boundary fraction: {boundary_fraction:.4%}")
        if boundary_fraction < 0.01:
            print("  -> Likely safe for aggregation (comparable to validated successes).")
        elif boundary_fraction < 0.10:
            print("  -> Untested range -- proceed but verify interior_coarse "
                  "is non-empty and substantial after aggregation.")
        else:
            print("  -> HIGH RISK. Comparable to or worse than confirmed "
                  "failures (Rhesus 25%, bio-CE-GN 70%). Consider a "
                  "different gamma-selection method, or (for diameter-"
                  "endpoint selection) a smaller k_hop relative to the "
                  "graph's diameter, before proceeding.")
    return boundary_fraction


# ============================================================
# Aggregation
# ============================================================

def build_capped_aggregation(G_nx, gamma_in, gamma_out, max_size=10):
    """
    Group fine vertices into aggregates (coarse vertices), respecting
    gamma_in/gamma_out as boundary conditions.

    Design choices, each motivated by a specific failure mode found
    during development:

    1. Boundary vertices (gamma_in/gamma_out) are processed FIRST, and
       each is forced to pair with an INTERIOR fine vertex specifically
       -- never with another boundary vertex, and never left to stand
       alone. Without this, boundary vertices can end up as their own
       singleton coarse vertices; if gamma_in+gamma_out is large enough
       relative to the graph, EVERY coarse vertex can end up being a
       boundary vertex, leaving zero interior coarse vertices for the
       Darcy solve (this is exactly what happened on Rhesus Brain).

    2. gamma_in vertices are never allowed to merge with gamma_out
       vertices (or vice versa) at any point -- a single coarse vertex
       cannot simultaneously be a fixed source (p=1) and fixed sink (p=0).

    3. Aggregate size is capped at max_size. Uncapped aggregation can
       let a few aggregates absorb many fine edges' permeability into
       one coarse edge, inflating Q_coarse by 10-100x relative to
       Q_fine even after normalization (observed on Rhesus Brain with
       uncapped aggregation).

    4. Remaining (non-boundary) vertices are processed in order of
       increasing degree, giving low-degree, few-option vertices first
       pick of their neighbors.

    Parameters
    ----------
    G_nx      : networkx.Graph
    gamma_in  : list of int
    gamma_out : list of int
    max_size  : int -- maximum fine vertices per aggregate

    Returns
    -------
    aggregate_of : dict {fine_vertex: coarse_vertex_id}
    n_coarse     : int -- number of coarse vertices (aggregates)
    """
    gamma_in_set = set(gamma_in)
    gamma_out_set = set(gamma_out)
    boundary_set = gamma_in_set | gamma_out_set

    aggregate_of = {}
    agg_sizes = {}
    agg_has_gamma_in = {}
    agg_has_gamma_out = {}
    next_agg_id = 0

    def forbidden_pair(a, b):
        return (a in gamma_in_set and b in gamma_out_set) or \
               (a in gamma_out_set and b in gamma_in_set)

    # Step 1: boundary vertices first, forced to pair with an interior neighbor
    boundary_nodes_by_degree = sorted(boundary_set, key=lambda v: G_nx.degree(v))
    for v in boundary_nodes_by_degree:
        if v in aggregate_of:
            continue
        partner = None
        for neighbor in G_nx.neighbors(v):
            if neighbor not in aggregate_of and neighbor not in boundary_set:
                partner = neighbor
                break
        if partner is not None:
            aggregate_of[v] = next_agg_id
            aggregate_of[partner] = next_agg_id
            agg_sizes[next_agg_id] = 2
            agg_has_gamma_in[next_agg_id] = v in gamma_in_set
            agg_has_gamma_out[next_agg_id] = v in gamma_out_set
            next_agg_id += 1
        # if no interior neighbor is available, v is picked up in Step 2

    # Step 2: everyone else (interior vertices, plus any leftover boundary vertices)
    remaining_by_degree = sorted(
        [v for v in G_nx.nodes() if v not in aggregate_of],
        key=lambda v: G_nx.degree(v)
    )
    for v in remaining_by_degree:
        if v in aggregate_of:
            continue
        partner = None
        for neighbor in G_nx.neighbors(v):
            if neighbor not in aggregate_of and not forbidden_pair(v, neighbor):
                partner = neighbor
                break
        if partner is not None:
            aggregate_of[v] = next_agg_id
            aggregate_of[partner] = next_agg_id
            agg_sizes[next_agg_id] = 2
            agg_has_gamma_in[next_agg_id] = (v in gamma_in_set) or (partner in gamma_in_set)
            agg_has_gamma_out[next_agg_id] = (v in gamma_out_set) or (partner in gamma_out_set)
            next_agg_id += 1
        else:
            joined = False
            for neighbor in G_nx.neighbors(v):
                if neighbor in aggregate_of and not forbidden_pair(v, neighbor):
                    agg_id = aggregate_of[neighbor]
                    v_is_in = v in gamma_in_set
                    v_is_out = v in gamma_out_set
                    if (v_is_in and agg_has_gamma_out.get(agg_id, False)) or \
                       (v_is_out and agg_has_gamma_in.get(agg_id, False)):
                        continue
                    if agg_sizes[agg_id] < max_size:
                        aggregate_of[v] = agg_id
                        agg_sizes[agg_id] += 1
                        agg_has_gamma_in[agg_id] = agg_has_gamma_in.get(agg_id, False) or v_is_in
                        agg_has_gamma_out[agg_id] = agg_has_gamma_out.get(agg_id, False) or v_is_out
                        joined = True
                        break
            if not joined:
                aggregate_of[v] = next_agg_id
                agg_sizes[next_agg_id] = 1
                agg_has_gamma_in[next_agg_id] = v in gamma_in_set
                agg_has_gamma_out[next_agg_id] = v in gamma_out_set
                next_agg_id += 1

    unique_ids = sorted(set(aggregate_of.values()))
    relabel = {old: new for new, old in enumerate(unique_ids)}
    aggregate_of = {v: relabel[a] for v, a in aggregate_of.items()}
    return aggregate_of, len(unique_ids)


def summarize_aggregation(aggregate_of, n_coarse, gamma_in, gamma_out, verbose=True):
    """
    Print/return diagnostic summary of an aggregation result: size
    distribution, singleton rate, and -- critically -- whether any
    interior coarse vertices remain and whether gamma_in/gamma_out
    overlap at the coarse level.

    Always call this after build_capped_aggregation() and before
    proceeding to build the coarse graph. If interior_coarse is empty,
    STOP -- the Darcy solve will be degenerate. See check_boundary_fraction()
    for how to diagnose why.

    Returns
    -------
    dict with keys: sizes, gamma_in_coarse, gamma_out_coarse,
    overlap, interior_coarse
    """
    groups = {}
    for v, a in aggregate_of.items():
        groups.setdefault(a, []).append(v)
    sizes = [len(m) for m in groups.values()]

    gamma_in_coarse = sorted(set(aggregate_of[v] for v in gamma_in))
    gamma_out_coarse = sorted(set(aggregate_of[v] for v in gamma_out))
    overlap = set(gamma_in_coarse) & set(gamma_out_coarse)

    boundary_coarse = set(gamma_in_coarse) | set(gamma_out_coarse)
    interior_coarse = [v for v in range(n_coarse) if v not in boundary_coarse]

    if verbose:
        print(f"n_coarse: {n_coarse} ({n_coarse} aggregates)")
        print(f"Size distribution -- min: {min(sizes)}, max: {max(sizes)}, "
              f"mean: {np.mean(sizes):.2f}")
        print(f"Singletons: {sizes.count(1)} ({sizes.count(1)/n_coarse*100:.1f}%)")
        print(f"gamma_in_coarse: {len(gamma_in_coarse)}, "
              f"gamma_out_coarse: {len(gamma_out_coarse)}")
        print(f"Overlap (must be empty): {overlap}")
        print(f"Interior coarse vertices: {len(interior_coarse)} "
              f"({len(interior_coarse)/n_coarse*100:.1f}%)")
        if len(interior_coarse) == 0:
            print("  !! ZERO INTERIOR VERTICES -- Darcy solve will be "
                  "degenerate. Do not proceed. See check_boundary_fraction().")

    return {
        'sizes': sizes,
        'gamma_in_coarse': gamma_in_coarse,
        'gamma_out_coarse': gamma_out_coarse,
        'overlap': overlap,
        'interior_coarse': interior_coarse,
    }


# ============================================================
# Coarse graph construction
# ============================================================

def build_coarse_graph_edges(edges, aggregate_of):
    """
    Build the coarse graph's edge list from a fine edge list and an
    aggregation. Every fine edge crossing between two DIFFERENT
    aggregates contributes to (or creates) a coarse edge; fine edges
    internal to a single aggregate are dropped.

    Parameters
    ----------
    edges        : list of (i, j) tuples -- fine graph edges
    aggregate_of : dict {fine_vertex: coarse_vertex_id}

    Returns
    -------
    coarse_edges   : list of (agg_i, agg_j) tuples, deduplicated
    coarse_contribs: dict {(agg_i, agg_j): [fine_edge_indices]} --
                     remembers which fine edges feed into each coarse
                     edge, needed every sample to derive coarse
                     permeability from that sample's fine permeability
    """
    coarse_contribs = {}
    for idx, (i, j) in enumerate(edges):
        ai, aj = aggregate_of[i], aggregate_of[j]
        if ai == aj:
            continue
        key = tuple(sorted([ai, aj]))
        coarse_contribs.setdefault(key, []).append(idx)
    coarse_edges = list(coarse_contribs.keys())
    return coarse_edges, coarse_contribs


class TwoLevelSetup:
    """
    Bundles everything needed for paired fine/coarse Monte Carlo
    sampling, built ONCE per dataset (never rebuilt per-sample).

    Construct via TwoLevelSetup.build(...), then pass the resulting
    object into run_one_paired_sample() / run_coarse_only_sample().
    """

    def __init__(self, edges, n_vertices, gamma_in, gamma_out,
                 coarse_edges, coarse_contribs, n_coarse,
                 gamma_in_coarse, gamma_out_coarse,
                 B, factor, lambda_min):
        self.edges = edges
        self.n_vertices = n_vertices
        self.n_coarse = n_coarse
        self.coarse_edges = coarse_edges
        self.coarse_contribs = coarse_contribs
        self.B = B
        self.factor = factor
        self.lambda_min = lambda_min

        edges_arr = np.array(edges)
        self.i_arr = edges_arr[:, 0]
        self.j_arr = edges_arr[:, 1]
        self.row_idx = np.concatenate([self.i_arr, self.j_arr, self.i_arr, self.j_arr])
        self.col_idx = np.concatenate([self.i_arr, self.j_arr, self.j_arr, self.i_arr])

        coarse_edges_arr = np.array(coarse_edges)
        self.i_arr_c = coarse_edges_arr[:, 0]
        self.j_arr_c = coarse_edges_arr[:, 1]
        self.row_idx_c = np.concatenate([self.i_arr_c, self.j_arr_c, self.i_arr_c, self.j_arr_c])
        self.col_idx_c = np.concatenate([self.i_arr_c, self.j_arr_c, self.j_arr_c, self.i_arr_c])

        self.gamma_in = gamma_in
        self.gamma_out = gamma_out
        self.gamma_out_set = set(gamma_out)
        self.boundary = set(gamma_in) | set(gamma_out)
        self.interior = np.array([v for v in range(n_vertices) if v not in self.boundary])
        self.p_known = np.full(n_vertices, np.nan)
        for v in gamma_in:
            self.p_known[v] = 1.0
        for v in gamma_out:
            self.p_known[v] = 0.0

        self.gamma_in_coarse = gamma_in_coarse
        self.gamma_out_coarse = gamma_out_coarse
        self.gamma_out_coarse_set = set(gamma_out_coarse)
        self.boundary_coarse = set(gamma_in_coarse) | set(gamma_out_coarse)
        self.interior_coarse = np.array([v for v in range(n_coarse) if v not in self.boundary_coarse])
        self.p_known_coarse = np.full(n_coarse, np.nan)
        for v in gamma_in_coarse:
            self.p_known_coarse[v] = 1.0
        for v in gamma_out_coarse:
            self.p_known_coarse[v] = 0.0

    @classmethod
    def build(cls, edges, n_vertices, L_sigma, lambda_min, gamma_in, gamma_out,
              build_incidence_matrix, sparse_cholesky, max_size=10, verbose=True):
        """
        Full one-time setup: build the aggregation, coarse graph, and
        all fixed structures needed for repeated paired sampling.

        Parameters
        ----------
        edges, n_vertices, L_sigma, lambda_min : standard Phase 1-2 outputs
        gamma_in, gamma_out       : from your chosen gamma-selection method
        build_incidence_matrix    : pass in from functions.py
        sparse_cholesky           : pass in `from sksparse.cholmod import cholesky as sparse_cholesky`
        max_size                  : aggregation cap, see build_capped_aggregation
        """
        check_boundary_fraction(gamma_in, gamma_out, n_vertices, verbose=verbose)

        G_nx = nx.Graph()
        G_nx.add_edges_from(edges)

        aggregate_of, n_coarse = build_capped_aggregation(G_nx, gamma_in, gamma_out, max_size=max_size)
        summary = summarize_aggregation(aggregate_of, n_coarse, gamma_in, gamma_out, verbose=verbose)

        if len(summary['interior_coarse']) == 0:
            raise ValueError(
                "Aggregation produced zero interior coarse vertices -- "
                "the Darcy solve would be degenerate. See "
                "check_boundary_fraction() output above for diagnosis."
            )
        if len(summary['overlap']) > 0:
            raise ValueError(
                f"gamma_in_coarse and gamma_out_coarse overlap at "
                f"{summary['overlap']} -- this should not happen given "
                f"build_capped_aggregation's boundary protection; check "
                f"for a bug or a modified aggregation function."
            )

        coarse_edges, coarse_contribs = build_coarse_graph_edges(edges, aggregate_of)
        if verbose:
            print(f"coarse edges: {len(coarse_edges)} (from {len(edges)} fine edges)")

        B = build_incidence_matrix(edges, n_vertices)
        factor = sparse_cholesky(L_sigma)

        return cls(edges, n_vertices, gamma_in, gamma_out,
                   coarse_edges, coarse_contribs, n_coarse,
                   summary['gamma_in_coarse'], summary['gamma_out_coarse'],
                   B, factor, lambda_min)


# ============================================================
# Per-sample functions
# ============================================================

def run_one_paired_sample(setup: TwoLevelSetup, seed):
    """
    Draw ONE white noise sample and compute BOTH Q_fine and Q_coarse
    from it. This pairing -- both quantities derived from the same
    random draw -- is what keeps them correlated across many samples,
    which is the entire basis for the two-level method's variance
    reduction.

    Parameters
    ----------
    setup : TwoLevelSetup -- from TwoLevelSetup.build(...)
    seed  : int -- RNG seed for this sample

    Returns
    -------
    Q_fine, Q_coarse : float, float
    """
    rng = np.random.RandomState(seed)
    w = rng.randn(len(setup.edges))

    # Fine pipeline (Phases 3-6)
    f = setup.B @ w
    u = setup.factor.solve_A(np.sqrt(setup.lambda_min) * f)
    k_vals = np.exp((u[setup.i_arr] + u[setup.j_arr]) / 2)

    diag_data = np.concatenate([k_vals, k_vals])
    offdiag_data = np.concatenate([-k_vals, -k_vals])
    data = np.concatenate([diag_data, offdiag_data])
    L_k = coo_matrix((data, (setup.row_idx, setup.col_idx)),
                      shape=(setup.n_vertices, setup.n_vertices)).tocsr()

    L_interior = L_k[setup.interior, :][:, setup.interior]
    rhs = -L_k[setup.interior, :][:, list(setup.boundary)] @ setup.p_known[list(setup.boundary)]
    p_interior = spsolve(L_interior, rhs)

    p = np.zeros(setup.n_vertices)
    p[setup.interior] = p_interior
    for v in setup.gamma_in:
        p[v] = 1.0
    for v in setup.gamma_out:
        p[v] = 0.0

    p_diff = np.abs(p[setup.i_arr] - p[setup.j_arr])
    outlet_mask = np.array([i in setup.gamma_out_set or j in setup.gamma_out_set
                             for i, j in setup.edges])
    Q_fine = np.sum(k_vals[outlet_mask] * p_diff[outlet_mask])

    # Coarse pipeline -- permeability derived from the SAME sample's k_vals
    Q_coarse = _solve_coarse(setup, k_vals)

    return Q_fine, Q_coarse


def run_coarse_only_sample(setup: TwoLevelSetup, seed):
    """
    Cheap sample: skips the fine Darcy solve, but still requires the
    fine Phase 3 white-noise solve (coarse permeability is derived
    from fine u). See module docstring / project notes for the
    honest caveat that this makes "coarse-only" less cheap than it
    sounds on this pipeline, since Phase 3 -- not the Darcy solve --
    is usually the dominant per-sample cost once sparse Cholesky is
    in use.
    """
    rng = np.random.RandomState(seed)
    w = rng.randn(len(setup.edges))

    f = setup.B @ w
    u = setup.factor.solve_A(np.sqrt(setup.lambda_min) * f)
    k_vals = np.exp((u[setup.i_arr] + u[setup.j_arr]) / 2)

    return _solve_coarse(setup, k_vals)


def _solve_coarse(setup: TwoLevelSetup, k_vals):
    """Internal: coarse Darcy solve given this sample's fine k_vals."""
    coarse_k_vals = np.array([
        sum(k_vals[idx] for idx in setup.coarse_contribs[e])
        for e in setup.coarse_edges
    ])

    diag_c = np.concatenate([coarse_k_vals, coarse_k_vals])
    offdiag_c = np.concatenate([-coarse_k_vals, -coarse_k_vals])
    data_c = np.concatenate([diag_c, offdiag_c])
    L_k_c = coo_matrix((data_c, (setup.row_idx_c, setup.col_idx_c)),
                        shape=(setup.n_coarse, setup.n_coarse)).tocsr()

    L_interior_c = L_k_c[setup.interior_coarse, :][:, setup.interior_coarse]
    rhs_c = -L_k_c[setup.interior_coarse, :][:, list(setup.boundary_coarse)] \
        @ setup.p_known_coarse[list(setup.boundary_coarse)]
    p_interior_c = spsolve(L_interior_c, rhs_c)

    p_c = np.zeros(setup.n_coarse)
    p_c[setup.interior_coarse] = p_interior_c
    for v in setup.gamma_in_coarse:
        p_c[v] = 1.0
    for v in setup.gamma_out_coarse:
        p_c[v] = 0.0

    p_diff_c = np.abs(p_c[setup.i_arr_c] - p_c[setup.j_arr_c])
    outlet_mask_c = np.array([i in setup.gamma_out_coarse_set or j in setup.gamma_out_coarse_set
                               for i, j in setup.coarse_edges])
    Q_coarse = np.sum(coarse_k_vals[outlet_mask_c] * p_diff_c[outlet_mask_c])
    return Q_coarse


# ============================================================
# Validation / estimator
# ============================================================
def run_paired_validation(setup: TwoLevelSetup, N=300, seed_offset=0, verbose=True):
    """
    Run N paired samples and report the two numbers that determine
    whether two-level MC is working on this dataset: correlation and
    variance reduction.

    Returns
    -------
    dict with keys: Q_fine_samples, Q_coarse_samples, diff_samples,
    correlation, variance_reduction
    """
    Q_fine_samples = np.zeros(N)
    Q_coarse_samples = np.zeros(N)

    with tqdm(total=N, desc="Paired samples", unit="sample") as pbar:
        for n in range(N):
            qf, qc = run_one_paired_sample(setup, seed=seed_offset + n)
            Q_fine_samples[n] = qf
            Q_coarse_samples[n] = qc
            pbar.update(1)
            pbar.set_postfix({
                "Q_fine": f"{Q_fine_samples[:n+1].mean():.4f}",
                "Q_coarse": f"{Q_coarse_samples[:n+1].mean():.4f}",
            })

    diff_samples = Q_fine_samples - Q_coarse_samples
    correlation = np.corrcoef(Q_fine_samples, Q_coarse_samples)[0, 1]
    variance_reduction = Q_fine_samples.var() / diff_samples.var()

    if verbose:
        print(f"\nN = {N} paired samples")
        print(f"Q_fine   : mean={Q_fine_samples.mean():.6f}  var={Q_fine_samples.var():.6f}")
        print(f"Q_coarse : mean={Q_coarse_samples.mean():.6f}  var={Q_coarse_samples.var():.6f}")
        print(f"Q_fine - Q_coarse : mean={diff_samples.mean():.6f}  var={diff_samples.var():.6f}")
        print(f"Correlation(Q_fine, Q_coarse): {correlation:.4f}")
        print(f"Variance reduction: {variance_reduction:.2f}x")

    return {
        'Q_fine_samples': Q_fine_samples,
        'Q_coarse_samples': Q_coarse_samples,
        'diff_samples': diff_samples,
        'correlation': correlation,
        'variance_reduction': variance_reduction,
    }

def two_level_estimate(setup: TwoLevelSetup, paired_result, N_coarse_only=2000,
                        seed_offset=100000, verbose=True):
    """
    Combine many cheap coarse-only samples with the (already-computed)
    paired correction term to produce the full two-level estimate of
    E[Q_fine].

    Returns
    -------
    dict with keys: estimate, coarse_base_mean, correction_mean
    """
    Q_coarse_only = np.zeros(N_coarse_only)

    with tqdm(total=N_coarse_only, desc="Coarse-only samples", unit="sample") as pbar:
        for n in range(N_coarse_only):
            Q_coarse_only[n] = run_coarse_only_sample(setup, seed=seed_offset + n)
            pbar.update(1)
            pbar.set_postfix({"Q_coarse": f"{Q_coarse_only[:n+1].mean():.4f}"})

    coarse_base_mean = Q_coarse_only.mean()
    correction_mean = paired_result['diff_samples'].mean()
    estimate = coarse_base_mean + correction_mean

    if verbose:
        print(f"\nCoarse-only base estimate (N={N_coarse_only}): {coarse_base_mean:.6f}")
        print(f"Correction term mean (paired samples): {correction_mean:.6f}")
        print(f"Two-level estimate of E[Q_fine]: {estimate:.6f}")
        print(f"Direct fine-only mean (for comparison): "
              f"{paired_result['Q_fine_samples'].mean():.6f}")

    return {
        'estimate': estimate,
        'coarse_base_mean': coarse_base_mean,
        'correction_mean': correction_mean,
    }


# 
def visualize_coarse_graph(setup, aggregate_of, title="Coarse Graph"):
    """
    Visualize the coarse graph, with node size reflecting aggregate size
    and color distinguishing gamma_in_coarse / gamma_out_coarse / interior.
    """
    G_coarse = nx.Graph()
    G_coarse.add_nodes_from(range(setup.n_coarse))
    G_coarse.add_edges_from(setup.coarse_edges)

    # aggregate sizes -> node size
    groups = {}
    for v, a in aggregate_of.items():
        groups.setdefault(a, []).append(v)
    sizes = {a: len(m) for a, m in groups.items()}
    node_sizes = [50 + sizes.get(a, 1) * 15 for a in G_coarse.nodes()]

    # color by role: gamma_in (source), gamma_out (sink), interior
    gamma_in_coarse_set = set(setup.gamma_in_coarse)
    gamma_out_coarse_set = set(setup.gamma_out_coarse)
    node_colors = []
    for a in G_coarse.nodes():
        if a in gamma_in_coarse_set:
            node_colors.append('#ff9e9e')   # source - red
        elif a in gamma_out_coarse_set:
            node_colors.append('#9ecbff')   # sink - blue
        else:
            node_colors.append('#c8c8c8')   # interior - gray

    fig, ax = plt.subplots(figsize=(10, 10))
    pos = nx.spring_layout(G_coarse, seed=7, k=0.3)

    nx.draw_networkx_edges(G_coarse, pos, ax=ax, width=0.6, edge_color='#999', alpha=0.5)
    nx.draw_networkx_nodes(G_coarse, pos, ax=ax, node_size=node_sizes,
                            node_color=node_colors, edgecolors='#333', linewidths=0.4)

    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color='#ff9e9e', label='gamma_in (source, p=1)'),
        mpatches.Patch(color='#9ecbff', label='gamma_out (sink, p=0)'),
        mpatches.Patch(color='#c8c8c8', label='interior'),
    ]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=10)

    ax.set_title(f"{title}\n{setup.n_coarse} coarse vertices, {len(setup.coarse_edges)} edges "
                 f"(node size = aggregate size)", fontsize=13)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_convergence(result, title="Convergence Analysis"):
    """
    Plot the running mean of Q_fine and Q_coarse vs. sample count, to
    visually confirm whether N samples is enough for a stable estimate.
    A flat line by the end of the plot indicates convergence; visible
    drift or large swings mean more samples are needed before trusting
    the mean.
    """
    Q_fine = result['Q_fine_samples']
    Q_coarse = result['Q_coarse_samples']
    N = len(Q_fine)

    running_mean_fine = np.cumsum(Q_fine) / np.arange(1, N + 1)
    running_mean_coarse = np.cumsum(Q_coarse) / np.arange(1, N + 1)
    running_mean_diff = np.cumsum(Q_fine - Q_coarse) / np.arange(1, N + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: running means of Q_fine and Q_coarse ---
    ax = axes[0]
    ax.plot(range(1, N + 1), running_mean_fine, label='Q_fine running mean', color='#2c7fb8', linewidth=1.5)
    ax.plot(range(1, N + 1), running_mean_coarse, label='Q_coarse running mean', color='#e6550d', linewidth=1.5)
    ax.set_xlabel('Sample count (N)')
    ax.set_ylabel('Running mean of Q')
    ax.set_title('Convergence of Q_fine and Q_coarse')
    ax.legend()
    ax.grid(alpha=0.3)

    # --- Right: running mean of the correction term (Q_fine - Q_coarse) ---
    ax = axes[1]
    ax.plot(range(1, N + 1), running_mean_diff, color='#31a354', linewidth=1.5)
    ax.axhline(running_mean_diff[-1], color='#888', linestyle='--', linewidth=1,
               label=f'Final value: {running_mean_diff[-1]:.4f}')
    ax.set_xlabel('Sample count (N)')
    ax.set_ylabel('Running mean of (Q_fine - Q_coarse)')
    ax.set_title('Convergence of the Correction Term')
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()



"""
visualize_fine_coarse
 
General-purpose fine-vs-coarse graph visualization, usable across any
dataset in the two-level MC project regardless of scale (validated on
graphs from 15 to 1000+ vertices). Automatically adapts node sizing,
edge width, and layout parameters based on graph size, since a single
fixed sizing scheme looks wrong across very different vertex counts
(this was discovered directly while building Facebook/Oregon figures
for the paper -- see project notes).
 
Usage
-----
from visualize_fine_coarse import visualize_fine_and_coarse
 
fig, axes = visualize_fine_and_coarse(
    G_nx, aggregate_of, n_coarse, coarse_edges,
    gamma_in_coarse, gamma_out_coarse,
    dataset_name="Power Grid",
    save_path="power_grid_fine_coarse.png"
)
"""
 
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
 
 
def visualize_fine_and_coarse(
    G_nx,
    aggregate_of,
    n_coarse,
    coarse_edges,
    gamma_in_coarse,
    gamma_out_coarse,
    dataset_name="Dataset",
    save_path=None,
    figsize=(18, 8),
):
    """
    Plot the fine graph and its coarse graph side by side, colored by
    boundary role (gamma_in / gamma_out / interior).
 
    Parameters
    ----------
    G_nx              : networkx.Graph -- the fine graph
    aggregate_of      : dict {fine_vertex: coarse_vertex_id}
    n_coarse          : int
    coarse_edges      : list of (agg_i, agg_j) tuples
    gamma_in_coarse   : list of int -- coarse vertex ids that are gamma_in
    gamma_out_coarse  : list of int -- coarse vertex ids that are gamma_out
    dataset_name      : str -- used in plot titles
    save_path         : str or None -- if given, saves the figure here
    figsize           : tuple -- overall figure size
 
    Returns
    -------
    fig, axes : the matplotlib figure and axes, for further tweaking
    """
    n_vertices = G_nx.number_of_nodes()
    n_edges = G_nx.number_of_edges()
 
    # ---- Adaptive sizing thresholds, tuned across 15- to 1000+-vertex tests ----
    if n_vertices <= 50:
        node_size_fine, edge_width_fine, edge_alpha_fine = 500, 1.5, 0.8
        layout_iters, show_labels = 50, True
    elif n_vertices <= 2000:
        node_size_fine, edge_width_fine, edge_alpha_fine = 40, 0.4, 0.4
        layout_iters, show_labels = 25, False
    else:
        node_size_fine, edge_width_fine, edge_alpha_fine = 3, 0.08, 0.2
        layout_iters, show_labels = 12, False
 
    groups = {}
    for v, a in aggregate_of.items():
        groups.setdefault(a, []).append(v)
    agg_sizes = {a: len(m) for a, m in groups.items()}
 
    gamma_in_set = set(gamma_in_coarse)
    gamma_out_set = set(gamma_out_coarse)
 
    node_colors_fine = []
    for v in G_nx.nodes():
        a = aggregate_of[v]
        if a in gamma_in_set:
            node_colors_fine.append('#ff9e9e')
        elif a in gamma_out_set:
            node_colors_fine.append('#9ecbff')
        else:
            node_colors_fine.append('#b0b0b0')
 
    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor='white')
 
    # ---- Fine graph ----
    ax = axes[0]
    pos_fine = nx.spring_layout(G_nx, seed=7, k=None, iterations=layout_iters)
    nx.draw_networkx_edges(G_nx, pos_fine, ax=ax, width=edge_width_fine,
                            edge_color='#999999', alpha=edge_alpha_fine)
    nx.draw_networkx_nodes(
        G_nx, pos_fine, ax=ax, node_size=node_size_fine,
        node_color=node_colors_fine,
        edgecolors='none' if n_vertices > 50 else '#1a1a1a',
        linewidths=0 if n_vertices > 50 else 1.2,
    )
    if show_labels:
        nx.draw_networkx_labels(G_nx, pos_fine, ax=ax, font_size=9, font_weight='bold')
    ax.set_title(f"{dataset_name} — Fine graph\n{n_vertices} vertices, {n_edges} edges", fontsize=12)
    ax.axis('off')
 
    # ---- Coarse graph ----
    ax = axes[1]
    G_coarse = nx.Graph()
    G_coarse.add_nodes_from(range(n_coarse))
    G_coarse.add_edges_from(coarse_edges)
 
    if n_coarse <= 50:
        node_size_coarse = [200 + agg_sizes.get(a, 1) * 120 for a in G_coarse.nodes()]
        edge_width_coarse, show_labels_coarse = 1.5, True
    elif n_coarse <= 2000:
        node_size_coarse = [10 + agg_sizes.get(a, 1) * 4 for a in G_coarse.nodes()]
        edge_width_coarse, show_labels_coarse = 0.4, False
    else:
        node_size_coarse = [2 + agg_sizes.get(a, 1) * 0.8 for a in G_coarse.nodes()]
        edge_width_coarse, show_labels_coarse = 0.08, False
 
    node_colors_coarse = []
    for a in G_coarse.nodes():
        if a in gamma_in_set:
            node_colors_coarse.append('#ff9e9e')
        elif a in gamma_out_set:
            node_colors_coarse.append('#9ecbff')
        else:
            node_colors_coarse.append('#7fb37f')
 
    pos_coarse = nx.spring_layout(G_coarse, seed=7, k=None, iterations=layout_iters)
    nx.draw_networkx_edges(G_coarse, pos_coarse, ax=ax, width=edge_width_coarse,
                            edge_color='#666666', alpha=0.5)
    nx.draw_networkx_nodes(
        G_coarse, pos_coarse, ax=ax, node_size=node_size_coarse,
        node_color=node_colors_coarse,
        edgecolors='none' if n_coarse > 50 else '#1a1a1a',
        linewidths=0 if n_coarse > 50 else 1.2,
    )
    if show_labels_coarse:
        labels = {a: f"{a}\n({agg_sizes.get(a, 1)})" for a in G_coarse.nodes()}
        nx.draw_networkx_labels(G_coarse, pos_coarse, labels=labels, ax=ax, font_size=8, font_weight='bold')
    ax.set_title(
        f"{dataset_name} — Coarse graph\n{n_coarse} vertices, {len(coarse_edges)} edges "
        f"({n_vertices / n_coarse:.1f}x reduction)",
        fontsize=12,
    )
    ax.axis('off')
 
    legend_patches = [
        mpatches.Patch(color='#ff9e9e', label='gamma_in'),
        mpatches.Patch(color='#9ecbff', label='gamma_out'),
        mpatches.Patch(color='#b0b0b0', label='interior (fine)'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.02))
 
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=140, bbox_inches='tight', facecolor='white')
        print(f"saved to {save_path}")
 
    return fig, axes