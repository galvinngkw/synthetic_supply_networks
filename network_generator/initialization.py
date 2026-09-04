"""Network initialization functions.

This module provides functions for:
- Generating heavy-tailed degree sequences (Burr distribution)
- Reordering degrees to create correlation structure
- Sampling adjacency matrices using the configuration model
- Generating lognormal strength fitnesses
- Populating edge weights based on degrees and strengths using four theta parameters
"""

import igraph as ig
import numpy as np
import torch
from scipy import optimize
from scipy.special import gammaln, polygamma
from scipy.stats import burr12

# -----------------------------------------------------------------------------
# Degree Sequence Generation
# -----------------------------------------------------------------------------


def obj_fn(
    k: float,
    target_alpha: float,
    target_log_var: float,
) -> float:
    c = target_alpha / k
    trigamma_k = polygamma(1, k)
    return ((np.pi**2 / 6.0 + trigamma_k) / (c**2)) - target_log_var


def get_parameters_for_burr_distribution(
    target_alpha_input: float,
    target_log_var_input: float,
    target_mean_input: float,
) -> tuple[float, float, float]:
    sol = optimize.root_scalar(
        obj_fn,
        args=(target_alpha_input, target_log_var_input),
        bracket=(0.01, 50.0),
        method="brentq",
    )
    k = sol.root
    c = target_alpha_input / k

    log_term1 = gammaln(k - 1.0 / c) + gammaln(1.0 + 1.0 / c) - gammaln(k + 1.0)
    term1 = np.exp(log_term1)

    lam = target_mean_input / (k * term1)
    return k, c, lam


def generate_burr_degrees(
    c_val: float, d_val: float, lam_val: float, max_cutoff: float, num_of_firms: int
) -> torch.Tensor:
    """Generate a truncated Burr degree sequence.

    Args:
        c_val: parameter input for the Burr distribution.
        d_val: parameter input for the Burr distribution.
        lam_val: parameter input for the Burr distribution
        max_cutoff: Maximum degree as fraction of num_of_firms.
        num_of_firms: Number of firms (samples to generate).

    Returns:
        Tensor of rounded degree values.
    """
    store = []
    while len(store) < num_of_firms:
        input_sample = burr12.rvs(
            c=c_val, d=d_val, scale=lam_val, size=num_of_firms - len(store)
        )
        truncate_input_sample = input_sample[
            input_sample < max_cutoff.item() * num_of_firms
        ]
        store += list(truncate_input_sample)
    return torch.tensor(store).round().to(torch.int64)


def generate_burr_degrees_matching_sum(
    c_val, d_val, lam_val, max_cutoff, num_of_firms, degree_sum
):
    seq = generate_burr_degrees(c_val, d_val, lam_val, max_cutoff, num_of_firms)
    difference = int((seq.sum() - degree_sum).item())
    while difference != 0:
        probs = seq.float() / seq.sum().float()
        ss = torch.multinomial(probs, num_samples=abs(difference), replacement=True)
        if difference > 0:
            seq -= torch.bincount(ss, minlength=num_of_firms)
        else:
            seq += torch.bincount(ss, minlength=num_of_firms)
        difference = int((seq.sum() - degree_sum).item())
    return seq


# -----------------------------------------------------------------------------
# Degree Reordering
# -----------------------------------------------------------------------------


def triangular_wedge_matching(
    in_deg_seq: torch.Tensor,
    out_deg_seq: torch.Tensor,
    noise_strength: float,
) -> torch.Tensor:
    """Reorder out-degrees to create correlation with in-degrees.

    Uses rank-based noise injection to create the characteristic
    "wedge" pattern observed in real supply networks.

    Args:
        in_deg_seq: In-degree sequence.
        out_deg_seq: Out-degree sequence to reorder.
        noise_strength: Controls correlation strength (-1 to 1).
        Higher positive values create more noise at lower ranks.

    Returns:
        Reordered out-degree sequence.
    """
    N = len(in_deg_seq)

    # Ensure positivity to avoid log(0)
    in_deg = in_deg_seq.clone()
    out_deg = out_deg_seq.clone()

    # argsort twice inverts the permutation to get ranks
    rank = torch.argsort(torch.argsort(in_deg))
    rank_norm = rank / (N - 1)

    # Apply lognormal noise: more noise at lower ranks
    noise_std = (1 - rank_norm) ** noise_strength
    noise = torch.randn(N) * noise_std

    # Unit mean lognormal noise
    lognorm_noise = torch.exp(noise - 0.5 * (noise_std**2))

    # Reorder out_deg to match scrambled in_deg ranking
    scrambled_out_deg = torch.zeros(N, dtype=out_deg.dtype)
    scrambled_out_deg[(in_deg * lognorm_noise).argsort()] = out_deg.sort().values

    return scrambled_out_deg


# -----------------------------------------------------------------------------
# Configuration Model
# -----------------------------------------------------------------------------


def directed_configuration_model(
    in_deg_seq: torch.Tensor,
    out_deg_seq: torch.Tensor,
) -> torch.Tensor:
    """Sample an adjacency matrix using the directed configuration model.

    Uses igraph's Degree_Sequence method with simplification to remove
    self-loops and parallel edges.

    Args:
        in_deg_seq: In-degree sequence (must sum to same value as out_deg_seq).
        out_deg_seq: Out-degree sequence.

    Returns:
        Sparse COO tensor representing the adjacency matrix.

    Raises:
        AssertionError: If degree sequences don't sum to the same value.
    """

    in_list = in_deg_seq.int().tolist()
    out_list = out_deg_seq.int().tolist()

    assert sum(in_list) == sum(out_list), "In-degree and out-degree sums must match"

    N = len(in_list)

    # Generate graph using configuration model
    g = ig.Graph.Degree_Sequence(out=out_list, in_=in_list, method="simple")

    # Remove parallel edges and self-loops
    g_simple = g.simplify(multiple=True, loops=True)

    # Convert to sparse tensor
    edges = g_simple.get_edgelist()
    indices = torch.tensor(edges).t()  # [2, E]
    values = torch.ones(indices.shape[1])

    # Verify no duplicates
    assert indices.size(1) == torch.unique(indices, dim=1).size(
        1
    ), "Duplicate edges detected"

    return torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()


# -----------------------------------------------------------------------------
# Generate lognormal strength fitnesses
# -----------------------------------------------------------------------------


def generate_initial_strength_fitnesses(
    mean_in_str: float,
    mean_out_str: float,
    var_in_str: float,
    var_out_str: float,
    correlation: float,
    num_of_firms: int,
    monetary_unit: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate correlated lognormal in/out strength fitnesses.

    Args:
        mean_in_str: Mean in-strength fitness in dollar units.
        mean_out_str: Mean out-strength fitness in dollar units.
        var_in_str: Variance of in-strength fitness.
        var_out_str: Variance of out-strength fitness.
        correlation: Target correlation between in and out strength fitnesses.
        num_of_firms: Number of firms.
        monetary_unit: Monetary unit for scaling.

    Returns:
        Tuple of (in_strength_seq, out_strength_seq).
    """

    # Compute log-space parameters

    in_str_log_std = torch.sqrt(torch.tensor(var_in_str))
    in_str_log_mean = torch.tensor([mean_in_str]) - torch.log(
        torch.tensor([monetary_unit])
    )

    out_str_log_std = torch.sqrt(torch.tensor(var_out_str))
    out_str_log_mean = torch.tensor([mean_out_str]) - torch.log(
        torch.tensor([monetary_unit])
    )

    # Generate in-strengths
    in_str_seq = torch.exp(in_str_log_std * torch.randn(num_of_firms) + in_str_log_mean)

    # Generate correlated out-strengths
    stdev = ((1 - (correlation**2)) * (out_str_log_std**2)) ** 0.5
    out_str_seq = torch.exp(
        stdev * torch.randn(num_of_firms)
        + out_str_log_mean
        + (correlation * out_str_log_std / in_str_log_std)
        * (in_str_seq.log() - in_str_seq.log().mean())
    )

    # Ensure sum(in_str) == sum(out_str)
    out_str_seq *= in_str_seq.sum() / out_str_seq.sum()

    return in_str_seq, out_str_seq


# -----------------------------------------------------------------------------
# Weight Population
# -----------------------------------------------------------------------------


def populate_weights_fourparams(
    in_deg_seq: torch.Tensor,
    out_deg_seq: torch.Tensor,
    in_str_seq: torch.Tensor,
    out_str_seq: torch.Tensor,
    adjacency_sparse: torch.Tensor,
    theta_k_in: float | torch.Tensor,
    theta_k_out: float | torch.Tensor,
    theta_f_in: float | torch.Tensor,
    theta_f_out: float | torch.Tensor,
) -> torch.Tensor:
    """Populate edge weights based on degrees and strengths.

    Computes weights as:
        W_ij = (f_i^out)^theta_f_out * (f_j^in)^theta_f_in * (k_i^out)^theta_k_out * (k_j^in)^theta_k_in

    Args:
        in_deg_seq: In-degree sequence.
        out_deg_seq: Out-degree sequence.
        in_str_seq: In-strength (total incoming weight) sequence.
        out_str_seq: Out-strength (total outgoing weight) sequence.
        adjacency_sparse: Sparse COO adjacency matrix.
        theta_k_in: Exponent for in-degree term (can be float or tensor).
        theta_k_out: Exponent for out-degree term (can be float or tensor).
        theta_f_in: Exponent for in-strength fitness term (can be float or tensor).
        theta_f_out: Exponent for out-strength fitness term (can be float or tensor).

    Returns:
        Sparse COO tensor with populated weights.
    """
    adjacency_sparse = adjacency_sparse.coalesce()
    row_idx, col_idx = adjacency_sparse.indices()

    # Ensure theta_k_in, theta_k_out, theta_f_in, theta_f_out are tensors for gradient tracking
    if isinstance(theta_k_in, torch.Tensor):
        theta_k_in_val = theta_k_in
    else:
        theta_k_in_val = torch.tensor(theta_k_in, dtype=in_deg_seq.dtype)

    if isinstance(theta_k_out, torch.Tensor):
        theta_k_out_val = theta_k_out
    else:
        theta_k_out_val = torch.tensor(theta_k_out, dtype=out_deg_seq.dtype)

    if isinstance(theta_f_in, torch.Tensor):
        theta_f_in_val = theta_f_in
    else:
        theta_f_in_val = torch.tensor(theta_f_in, dtype=in_deg_seq.dtype)

    if isinstance(theta_f_out, torch.Tensor):
        theta_f_out_val = theta_f_out
    else:
        theta_f_out_val = torch.tensor(theta_f_out, dtype=in_deg_seq.dtype)

    ki_pow = torch.pow(in_deg_seq[col_idx], theta_k_in_val)
    si_pow = torch.pow(in_str_seq[col_idx], theta_f_in_val)
    ko_pow = torch.pow(out_deg_seq[row_idx], theta_k_out_val)
    so_pow = torch.pow(out_str_seq[row_idx], theta_f_out_val)

    weights = so_pow * si_pow * ki_pow * ko_pow

    return torch.sparse_coo_tensor(
        adjacency_sparse.indices(),
        weights,
        size=(in_deg_seq.numel(), in_deg_seq.numel()),
    )
