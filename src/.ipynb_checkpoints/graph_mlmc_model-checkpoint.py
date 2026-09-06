"""
graph_mlmc_model.py

Adapts the graph-based white-noise-sampling two-level pipeline
(functions_v2.py, two_level_mc.py) to the team's MultilevelModel
protocol (mlmc_model.py), so both the graph-based approach and the
PDE-based approach can be run through the same MLMCRunner and compared
directly using the same statistics, reproducibility, and reporting
machinery.

-------------------------------------------------------------------
WHAT THIS FILE DOES NOT CHANGE
-------------------------------------------------------------------
No graph-domain logic is altered. Aggregation (build_capped_aggregation),
the boundary-fraction precondition, and the sparse-Cholesky solve are
all preserved exactly as validated in two_level_mc.py / functions_v2.py.
This file is purely an adapter layer -- it re-packages existing
functions to satisfy the MultilevelModel Protocol's four methods.

-------------------------------------------------------------------
TWO THINGS THIS ADAPTER DELIBERATELY WORKS AROUND, WORTH DISCUSSING
WITH THE TEAM BEFORE TREATING THIS AS A FINAL MERGE:
-------------------------------------------------------------------
1. Sparse Cholesky with cached factorization is NOT available through
   their linear_solver.py (only dense `solve` / sparse `spsolve` via
   LU). The white-noise solve (Phase 3) is therefore performed here,
   inside couple_inputs(), using the existing scikit-sparse factor
   directly -- OUTSIDE their solve_linear_system() call path. Only
   the Darcy flow solve (fine and coarse) goes through their solver.
   This preserves the ~500x validated speedup on large graphs, but
   means this adapter does not exercise their solver abstraction for
   every linear solve in the pipeline, only the Darcy step.

2. number_of_levels is fixed at 2 (level 0 = coarse, level 1 = fine).
   The aggregation logic could in principle be applied recursively to
   build a true multi-level hierarchy (coarsen the coarse graph again,
   and again), which their framework already supports structurally --
   this adapter does not attempt that extension.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix

from linear_solver import LinearSystem
from mlmc_model import CoupledInputs

# --- existing, unmodified project code ---
from two_level_mc import TwoLevelSetup


# ============================================================
# Model input: everything build_linear_system / quantity_of_interest
# need for ONE level (fine or coarse), for ONE sample
# ============================================================

@dataclass(frozen=True)
class GraphModelInput:
     """
    One level's sample-dependent data, bundled for a single Monte
    Carlo sample.
 
    This is the object returned inside CoupledInputs by
    GraphTwoLevelModel.couple_inputs(), and consumed by
    GraphTwoLevelModel.build_linear_system() and
    GraphTwoLevelModel.quantity_of_interest(). It exists purely to
    satisfy the MultilevelModel protocol's expectation that
    couple_inputs() returns a single "model_input" object per level --
    everything a level needs to build its system and extract Q is
    packaged here rather than passed as separate arguments.
 
    All fields except `k_vals` are FIXED, graph-structural quantities
    that never change between samples -- they are copied here from
    the corresponding TwoLevelSetup / CoarsePipeline attributes purely
    so that build_linear_system() and quantity_of_interest() can
    remain pure functions of (level, model_input), as the protocol
    requires, without needing separate access to the setup object.
 
    Attributes
    ----------
    k_vals : np.ndarray, shape (n_edges_at_this_level,)
        This sample's edge permeability values for this level (fine
        or coarse), in the same order as `edges`. This is the ONLY
        field that is genuinely fresh every sample -- everything else is
        fixed structure carried along for convenience.
    edges : list of (int, int)
        Edge list for this level (setup.edges for fine,
        setup.coarse_edges for coarse).
    n_vertices : int
        Number of vertices at this level (setup.n_vertices for fine,
        setup.n_coarse for coarse).
    interior : np.ndarray of int
        Interior vertex indices at this level (vertices not in
        gamma_in or gamma_out at this level).
    boundary : list of int
        Boundary vertex indices at this level (gamma_in and gamma_out
        combined), as a plain list for use as a fancy-index into
        sparse matrix slices.
    p_known : np.ndarray, shape (n_vertices,)
        Fixed pressure lookup array for this level: p_in at gamma_in
        positions, p_out at gamma_out positions, NaN elsewhere
        (interior positions are never read from this array -- they
        are solved for, not looked up).
    i_arr, j_arr : np.ndarray of int
        Edge endpoint index arrays for this level, i.e. edges[:,0]
        and edges[:,1] pulled out once as separate arrays for
        vectorized indexing.
    row_idx, col_idx : np.ndarray of int
        Precomputed four-position-per-edge COO index arrays for this
        level, used to build the weighted Laplacian L_k every sample.
        This pattern depends only on graph structure (which edges
        exist), never on that sample's permeability values, so it is
        computed once (in TwoLevelSetup / CoarsePipeline) and reused
        here rather than rebuilt per sample.
    gamma_out_set : set of int
        Sink vertex set at this level, as a Python set, for fast
        membership checks when identifying which edges touch the
        outlet boundary in quantity_of_interest().
    """

    k_vals: np.ndarray          # this sample's edge permeabilities (fine or coarse)
    edges: list                 # edge list for this level
    n_vertices: int
    interior: np.ndarray
    boundary: list
    p_known: np.ndarray
    i_arr: np.ndarray
    j_arr: np.ndarray
    row_idx: np.ndarray
    col_idx: np.ndarray
    gamma_out_set: set


# ============================================================
# The adapter itself
# ============================================================

class GraphTwoLevelModel:
    """
    Adapts an existing, already-validated TwoLevelSetup (built via
    TwoLevelSetup.build(), exactly as validated across US Power Grid,
    Oregon Router, bio-grid-yeast, C. elegans, and Facebook Ego
    Network) to the team's MultilevelModel Protocol, so it can be
    driven by their generic MLMCRunner.
 
    This class contains NO new mathematics. Every method's body calls
    the same functions and formulas already validated in
    functions_v2.py / two_level_mc.py; this class only exists to
    present that logic through the five specific method names and
    signatures the MultilevelModel protocol requires
    (number_of_levels, sample_randomness, couple_inputs,
    build_linear_system, quantity_of_interest), and to translate
    between this project's data structures (TwoLevelSetup) and the
    team's (CoupledInputs, LinearSystem).
 
    Usage
    -----
    >>> setup = TwoLevelSetup.build(
    ...     edges, n_vertices, L_sigma, lambda_min,
    ...     gamma_in, gamma_out,
    ...     build_incidence_matrix, sparse_cholesky,
    ...     max_size=10,
    ... )
    >>> model = GraphTwoLevelModel(setup)
    >>> runner = MLMCRunner(model, base_seed=0)
    >>> result = runner.run_fixed(samples_per_level=[2000, 300])
    >>> print(result.estimate, result.standard_error)
 
    Parameters
    ----------
    setup : TwoLevelSetup
        An already-built, already-validated two-level setup for one
        dataset -- i.e. the output of TwoLevelSetup.build(), which has
        already run the boundary-fraction check, the aggregation
        algorithm, and the interior/overlap validity checks. This
        adapter does not repeat or re-validate any of that; it assumes
        `setup` is already known-good.
    """

    def __init__(self, setup: TwoLevelSetup):
        self.setup = setup

    # ------------------------------------------------------------
    # 1. number_of_levels
    # ------------------------------------------------------------
    @property
    def number_of_levels(self) -> int:
        """
        Report the number of levels this model supports, as required
        by the MultilevelModel protocol.
 
        Fixed at 2 for this adapter: level 0 is the coarse graph,
        level 1 is the fine graph. See the module docstring's "known
        gaps" section for discussion of extending this to a true
        multi-level hierarchy via recursive coarsening -- not
        attempted here.
 
        Returns
        -------
        int
            Always 2.
        """
        return 2

    # ------------------------------------------------------------
    # 2. sample_randomness
    # ------------------------------------------------------------
    def sample_randomness(
        self,
        fine_level: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """"
        Draw the shared randomness for one MLMC correction sample, as
        required by the MultilevelModel protocol.
 
        This is Phase 3a of the pipeline (functions_v2.py): draw
        w_e ~ N(0,1) independently for every FINE edge, regardless of
        which level was requested. This single draw is what
        couple_inputs() uses to derive BOTH the fine and coarse
        permeability fields for this sample, which is what keeps
        Q_fine and Q_coarse correlated -- the entire statistical basis
        for the two-level method's variance reduction.
 
        Parameters
        ----------
        fine_level : int
            The level this sample's correction is being computed for
            (0 or 1). Present because the protocol's signature
            requires it, but unused here -- the randomness drawn is
            always on the fine graph's edges regardless of level,
            since couple_inputs() derives the coarse quantities from
            the fine draw rather than sampling independently.
        rng : np.random.Generator
            Random number generator supplied by the team's MLMCRunner.
            Per the protocol's contract, this generator (not an
            independently-seeded one) is used, so that MLMCRunner's
            reproducible, collision-safe seeding scheme
            (SeedSequence([base_seed, level, sample_index])) governs
            this draw.
 
        Returns
        -------
        np.ndarray, shape (n_edges,)
            w, the fine graph's edge noise vector for this sample.
        """
        return rng.standard_normal(len(self.setup.edges))

    # ------------------------------------------------------------
    # 3. couple_inputs
    # ------------------------------------------------------------
    def couple_inputs(
        self,
        fine_level: int,
        randomness: np.ndarray,
    ) -> CoupledInputs[GraphModelInput]:
        """
        Construct adjacent-level inputs from one random realization,
        as required by the MultilevelModel protocol.
 
        Solves the fine white-noise system ONCE (Phase 3b-3c of the
        pipeline: f = Bw, then L_sigma u = sqrt(lambda_min) f via the
        cached sparse Cholesky factor), derives fine edge permeability
        (Phase 4: k_e = exp((u_i+u_j)/2)), and ALWAYS additionally
        derives coarse edge permeability from that same sample's fine
        k_vals (by summing, for each coarse edge, the fine k_vals of
        every fine edge that was absorbed into it during aggregation --
        see TwoLevelSetup.coarse_contribs). Deriving coarse
        permeability this way, rather than sampling it independently,
        is what keeps Q_fine and Q_coarse correlated.
 
        LEVEL-0 CONVENTION (see module docstring for the full story of
        the bug this fixed): at fine_level == 0, the coarsest level has
        no level below it to form a correction against, so the
        protocol's convention is that the "fine" slot returned here
        must itself represent the COARSE graph's input -- not the
        actual fine graph. This is why `coarse_input` is always built
        first, and only returned as `fine` when fine_level == 0. At
        fine_level == 1, both the true fine input and the coarse input
        are returned, giving the Q_fine - Q_coarse correction.
 
        NOTE on solver choice: the white-noise solve above uses
        `self.setup.factor.solve_A(...)` directly -- the cached sparse
        Cholesky (CHOLMOD) factorization built once in
        TwoLevelSetup.build() -- rather than routing through the
        team's solve_linear_system(). This is a deliberate choice: the
        team's linear_solver.py has no cached-factorization option, and
        refactorizing L_sigma from scratch on every sample would lose
        the ~500x speedup validated in this project (see
        functions_v2.py). Only the Darcy flow solve (in
        build_linear_system(), called separately by the team's
        framework) goes through their generic solver.
 
        Parameters
        ----------
        fine_level : int
            The level this correction is being computed for (0 or 1).
        randomness : np.ndarray, shape (n_edges,)
            The fine graph's white noise vector w, from
            sample_randomness().
 
        Returns
        -------
        CoupledInputs[GraphModelInput]
            `coarse` is None if fine_level == 0 (no level below the
            coarsest level to correct against), matching the
            protocol's required contract. `fine` is the coarse input
            if fine_level == 0, or the true fine input if
            fine_level == 1 -- see the level-0 convention note above.
        """
        setup = self.setup
        w = randomness
    
        f = setup.B @ w
        u = setup.factor.solve_A(np.sqrt(setup.lambda_min) * f)
        k_vals_fine = np.exp((u[setup.i_arr] + u[setup.j_arr]) / 2)
    
         # Coarse permeability is always derived -- cheap once u is
        # known (a sum-and-lookup over the fixed coarse_contribs
        # mapping, no additional linear solve), and needed regardless
        # of which level was requested, per the level-0 convention
        # discussed above.
        coarse_k_vals = np.array([
            sum(k_vals_fine[idx] for idx in setup.coarse_contribs[e])
            for e in setup.coarse_edges
        ])
        coarse_input = GraphModelInput(k_vals=coarse_k_vals, edges=setup.coarse_edges,
                                         n_vertices=setup.n_coarse, interior=setup.interior_coarse,
                                         boundary=list(setup.boundary_coarse), p_known=setup.p_known_coarse,
                                         i_arr=setup.i_arr_c, j_arr=setup.j_arr_c,
                                         row_idx=setup.row_idx_c, col_idx=setup.col_idx_c,
                                         gamma_out_set=setup.gamma_out_coarse_set)
    
        if fine_level == 0:
            # Coarsest level: the "fine" slot IS the coarse graph --
            # see the level-0 convention discussion above and in the
            # module docstring.
            return CoupledInputs(fine=coarse_input, coarse=None)
    
        # fine_level == 1: correction is fine minus coarse
        fine_input = GraphModelInput(k_vals=k_vals_fine, edges=setup.edges, n_vertices=setup.n_vertices,
                                       interior=setup.interior, boundary=list(setup.boundary),
                                       p_known=setup.p_known, i_arr=setup.i_arr, j_arr=setup.j_arr,
                                       row_idx=setup.row_idx, col_idx=setup.col_idx,
                                       gamma_out_set=setup.gamma_out_set)
        return CoupledInputs(fine=fine_input, coarse=coarse_input)

    # ------------------------------------------------------------
    # 4. build_linear_system
    # ------------------------------------------------------------
    def build_linear_system(
        self,
        level: int,
        model_input: GraphModelInput,
    ) -> LinearSystem:
        """
        Build one sample-dependent linear system at one level, as
        required by the MultilevelModel protocol.
 
        This is Phase 5 of the pipeline (functions_v2.py): build the
        permeability-weighted Laplacian L_k via the fixed
        four-position-per-edge COO construction (using
        model_input.row_idx / col_idx, which never change between
        samples), then form the reduced interior/boundary Darcy system
        L_ii p_i = -L_ib p_b, matching solve_darcy()'s setup exactly.
        Unlike the white-noise solve in couple_inputs(), this step
        genuinely has nothing to cache -- L_k's entries are different
        every sample (they depend on that sample's permeability
        values), so there is no factorization to reuse here regardless
        of which solver ultimately performs the solve. This is why
        this step, unlike the white-noise solve, is left to go through
        the team's generic solve_linear_system() rather than being
        special-cased.
 
        Parameters
        ----------
        level : int
            The level this system is being built for (0 for coarse,
            1 for fine). Not used directly in this method's body,
            since all level-specific structure needed is already
            present on `model_input` -- but required by the protocol's
            signature, and useful for any future debugging/logging
            that wants to distinguish fine-level from coarse-level
            solves.
        model_input : GraphModelInput
            The level-specific input returned by couple_inputs() for
            this same sample -- provides this sample's k_vals plus the
            fixed structural arrays (row_idx, col_idx, interior,
            boundary, p_known) needed to assemble the system.
 
        Returns
        -------
        LinearSystem
            `A` is L_ii (the interior-interior block of L_k, sparse),
            `b` is the right-hand side -L_ib @ p_known[boundary]. The
            team's solve_linear_system() (or any SystemSolver passed
            into MLMCRunner) is expected to solve this for p_i, the
            interior pressure values.
        """
        k_vals = model_input.k_vals
        diag_data = np.concatenate([k_vals, k_vals])
        offdiag_data = np.concatenate([-k_vals, -k_vals])
        data = np.concatenate([diag_data, offdiag_data])

        L_k = coo_matrix(
            (data, (model_input.row_idx, model_input.col_idx)),
            shape=(model_input.n_vertices, model_input.n_vertices),
        ).tocsr()

        L_interior = L_k[model_input.interior, :][:, model_input.interior]
        rhs = -L_k[model_input.interior, :][:, model_input.boundary] \
            @ model_input.p_known[model_input.boundary]

        return LinearSystem(A=L_interior, b=rhs)

    # ------------------------------------------------------------
    # 5. quantity_of_interest
    # ------------------------------------------------------------
    def quantity_of_interest(
        self,
        level: int,
        solution: np.ndarray,
        model_input: GraphModelInput,
    ) -> float:
        """
        Calculate the scalar quantity of interest for one solved
        system, as required by the MultilevelModel protocol.
 
        This is Phase 5b (pressure field reassembly) followed by
        Phase 6 (Q extraction) of the pipeline: given the solved
        interior pressure values, reconstruct the full pressure field
        by combining them with the fixed boundary values from
        model_input.p_known, then compute
        Q = sum of k_e * |p_i - p_j| over every edge touching the sink
        set (model_input.gamma_out_set), matching extract_qoi()
        exactly.
 
        Parameters
        ----------
        level : int
            The level this solution was computed at (0 or 1). Not
            used directly in this method's body, for the same reason
            as in build_linear_system() -- all necessary level-specific
            structure is already present on `model_input`.
        solution : np.ndarray, shape (n_interior,)
            The solved interior pressure values p_i, as returned by
            whichever SystemSolver processed the LinearSystem built by
            build_linear_system() for this same sample and level.
        model_input : GraphModelInput
            The same level-specific input used to build the system
            that produced `solution` -- provides this sample's k_vals,
            the boundary/interior index arrays, and gamma_out_set
            needed to reassemble p and compute Q.
 
        Returns
        -------
        float
            Q for this sample and level: either Q_fine or Q_coarse,
            depending on which level's model_input was passed in. The
            team's SampleCorrection machinery combines this with the
            paired level's Q (if any) into the final correction value
            fine_qoi - coarse_qoi, or just fine_qoi at level 0 per the
            level-0 convention discussed in couple_inputs().
        """
        p = np.zeros(model_input.n_vertices)
        p[model_input.interior] = solution
        boundary_indices = np.array(model_input.boundary)
        p[boundary_indices] = model_input.p_known[boundary_indices]

        p_diff = np.abs(p[model_input.i_arr] - p[model_input.j_arr])
        outlet_mask = np.array([
            i in model_input.gamma_out_set or j in model_input.gamma_out_set
            for i, j in model_input.edges
        ])

        return float(np.sum(model_input.k_vals[outlet_mask] * p_diff[outlet_mask]))
