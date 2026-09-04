"""Functions to compute degrees, strengths, strengths, hill exponent, OLS, TLS, Pagerank, mean, variance, correlations, aggregated model percentiles"""

from math import ceil, floor

import igraph as ig
import torch
import torch.nn as nn

criterion_mse = nn.MSELoss()

# -----------------------------------------------------------------------------
# Custom Autograd Functions
# -----------------------------------------------------------------------------


class BinaryThreshold(torch.autograd.Function):
    """Straight-Through Estimator for binary thresholding.

    Forward: Returns 1 if input > 0, else 0.
    Backward: Passes gradient unchanged (identity).
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


# -----------------------------------------------------------------------------
# Compute degrees from sparse network model
# -----------------------------------------------------------------------------


def compute_degrees_from_model(
    model: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract in-degree and out-degree sequences from a sparse model.


    If you ever want to add edge-level features, use Straight-Through Estimator for differentiability.
    Otherwise, the default binarize function is used.

    Args:
        model: Sparse COO tensor representing the network.

    Returns:
        Tuple of (in_deg_seq, out_deg_seq).
    """
    model = model.coalesce()
    row_idx, col_idx = model.indices()
    edge_scores = model.values()
    num_firms = model.size(0)

    # If you ever want to add edge-level features, then use theBinary threshold with STE
    # adj_values = BinaryThreshold.apply(edge_scores)

    # binarize edge_scores
    adj_values = (edge_scores > 0).float()

    # Remove self-loops
    mask = row_idx != col_idx
    row_idx = row_idx[mask]
    col_idx = col_idx[mask]
    adj_values = adj_values[mask]

    # Compute degrees
    out_deg = torch.zeros(num_firms)
    out_deg.index_add_(0, row_idx, adj_values)

    in_deg = torch.zeros(num_firms)
    in_deg.index_add_(0, col_idx, adj_values)

    return in_deg, out_deg


class CustomReLU(torch.autograd.Function):
    """Modified ReLU with non-standard backward pass.

    Forward: Standard ReLU (max(0, x)).
    Backward: Uses -input for negative values instead of 0.

    This helps gradients flow through negative regions during optimization.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return torch.relu(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (input,) = ctx.saved_tensors
        grad_input = torch.where(input > 0, torch.ones_like(input), -input)
        return grad_output * grad_input


# -----------------------------------------------------------------------------
# Compute strengths from sparse network model
# -----------------------------------------------------------------------------


def compute_strengths_from_model(
    model: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract non-negative weights and strength sequences from a sparse model.

    If you ever want to add edge-level features, then use CustomReLU for differentiability through negative values.
    Otherweise, use the edge_scores directly because they are always non-negative.

    Args:
        model: Sparse COO tensor representing the network.

    Returns:
        Tuple of (non_neg_model, edge_weights, in_str_seq, out_str_seq).
    """
    model = model.coalesce()
    row_idx, col_idx = model.indices()
    edge_scores = model.values()
    num_firms = model.size(0)

    # If you ever want to add edge-level features, then use CustomReLU for non-negative weights
    # edge_weights = CustomReLU.apply(edge_scores).clone()

    # Apply ReLU to ensure non-negative weights as a precaution
    assert (edge_scores >= 0).all(), "there are negative edge weights in the model"
    edge_weights = edge_scores.clone()

    # Remove self-loops
    mask = row_idx != col_idx
    new_row_idx = row_idx[mask]
    new_col_idx = col_idx[mask]
    new_edge_weights = edge_weights[mask]

    # Build non-negative sparse model
    non_neg_model = torch.sparse_coo_tensor(
        torch.stack((new_row_idx, new_col_idx), dim=0),
        new_edge_weights,
        (num_firms, num_firms),
    ).coalesce()

    # Compute strengths
    out_str = torch.zeros(num_firms)
    out_str.index_add_(0, new_row_idx, new_edge_weights)

    in_str = torch.zeros(num_firms)
    in_str.index_add_(0, new_col_idx, new_edge_weights)

    return non_neg_model, new_edge_weights, in_str, out_str


# -----------------------------------------------------------------------------
# Network Metrics
# -----------------------------------------------------------------------------


def compute_quantiles(
    input: torch.Tensor,
    q: float,
    dim: int | None = None,
    keepdim: bool = False,
    interpolation: str = "nearest",
) -> torch.Tensor:
    """Optimized quantile computation using torch.kthvalue.

    Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592)
        - Much faster on large input sizes

    Args:
        input: Input tensor.
        q: Quantile value (scalar, 0 <= q <= 1).
        dim: Dimension to reduce. If None, flattens input.
        keepdim: Whether to keep the reduced dimension.
        interpolation: One of "nearest", "lower", "higher".

    Returns:
        Quantile value(s) as a tensor.
    """
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is supported (got {q})!")

    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            f"Supported interpolations are {{'nearest', 'lower', 'higher'}} "
            f"(got '{interpolation}')!"
        )

    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True)[0]

    if keepdim:
        return out

    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)


def compute_hill_exponent(
    distribution: torch.Tensor,
    threshold_quantile: float,
    is_degree: bool = False,
) -> torch.Tensor:
    """Estimate the tail exponent of a complementary cumulative distribution function (CCDF).

    For a power-law tail, the PDF and CCDF exponents differ by one:
        p(x) ~ x^(-alpha)          (probability density function)
        P(X >= x) ~ x^(-(alpha-1)) (complementary cumulative distribution function)

    Clauset et al. 2009, Eq. 3.1 gives the Hill estimator for the PDF exponent alpha:
        alpha = 1 + n * [sum_i^n log(x_i / x_min)]^-1.

    This function returns the CCDF exponent, alpha - 1, by omitting the leading 1:
        alpha - 1 = n * [sum_i^n log(x_i / x_min)]^-1.

    Args:
        distribution: Input distribution tensor.
        threshold_quantile: Quantile threshold for x_min.
        is_degree: If True, adjusts x_min by -0.5 (for discrete degrees).

    Returns:
        CCDF tail exponent (alpha - 1) as a scalar tensor.
    """
    threshold = compute_quantiles(distribution.to(torch.float32), threshold_quantile)

    large_values = distribution[distribution >= threshold]

    if is_degree:
        log_ratios = torch.log(large_values / (threshold - 0.5))
    else:
        log_ratios = torch.log(large_values / threshold)

    n = log_ratios.shape[0]
    hill_exp = n * (1 / log_ratios.sum())

    return hill_exp


def compute_ols_beta(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute OLS regression coefficient (slope) in differentiable form.

    Args:
        x: Independent variable.
        y: Dependent variable.

    Returns:
        Regression coefficient beta.
    """
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    cov_xy = ((x - mean_x) * (y - mean_y)).sum() / (len(x) - 1)
    return cov_xy / torch.var(x)


def compute_tls_slope(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute Total Least Squares (orthogonal regression) slope via SVD.

    Args:
        x: Independent variable.
        y: Dependent variable.

    Returns:
        TLS slope as a scalar tensor.
    """
    x_centered = x - x.mean()
    y_centered = y - y.mean()

    A = torch.stack([x_centered, y_centered], dim=1)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)

    v = Vh.T
    slope = v[1, 0] / v[0, 0]

    return slope


def compute_pagerank(
    model: torch.Tensor,
    damping: float = 0.5,
) -> torch.Tensor:
    """Compute PageRank influence vector using igraph.

    Args:
        model: Sparse COO tensor representing the network.
        damping: Damping factor for PageRank (default 0.5).

    Returns:
        PageRank values as a tensor.
    """
    model = model.coalesce()
    row_idx, col_idx = model.indices()
    edge_weights = model.values()

    row_list = row_idx.detach().tolist()
    col_list = col_idx.detach().tolist()
    weight_list = edge_weights.detach().tolist()

    g = ig.Graph(
        n=int(model.size(0)),
        edges=list(zip(col_list, row_list)),
        directed=True,
    )
    pagerank = g.pagerank(directed=True, damping=damping, weights=weight_list)

    return torch.tensor(pagerank, dtype=torch.float32)


def compute_supplieronly_customeronly_from_model(
    model: torch.Tensor,
) -> tuple[float, float]:
    """Compute the percentage of supplier-only and customer-only firms in the network."""
    model = model.coalesce()
    row_idx, col_idx = model.indices()
    num_firms = model.size(0)
    num_edges = model._nnz()

    # Build igraph representations
    edges = list(zip(row_idx.numpy(), col_idx.numpy()))

    # Directed graph for reciprocity
    ig_dir = ig.Graph(n=num_firms, edges=edges, directed=True)
    ig_dir.simplify(multiple=True, loops=True)

    # Giant component
    components = ig_dir.components(mode="WEAK")
    giant = components.giant()

    # In-degrees and out-degrees within the giant component
    in_deg = giant.degree(mode="in")
    out_deg = giant.degree(mode="out")
    n_giant = giant.vcount()

    # Supplier-only: in-degree 0; customer-only: out-degree 0 (percent of giant)
    supplier_only_pct = 100 * sum(1 for d in in_deg if d == 0) / n_giant
    customer_only_pct = 100 * sum(1 for d in out_deg if d == 0) / n_giant

    return supplier_only_pct, customer_only_pct


def compute_mean_and_variance_of_log_distributions(
    in_deg_seq: torch.Tensor,
    out_deg_seq: torch.Tensor,
    in_str_seq: torch.Tensor,
    out_str_seq: torch.Tensor,
    monetary_unit: float,
) -> dict[str, float]:
    """Compute mean and variance of log distributions.

    Args:
        in_deg_seq: In-degree sequence.
        out_deg_seq: Out-degree sequence.
        in_str_seq: In-strength sequence.
        out_str_seq: Out-strength sequence.
        monetary_unit: Monetary unit for scaling.
    """
    monetary_unit = torch.tensor(monetary_unit, dtype=torch.float32)
    return {
        "mean_log_in_deg": torch.mean(torch.log(in_deg_seq)).item(),
        "var_log_in_deg": torch.var(torch.log(in_deg_seq)).item(),
        "mean_log_out_deg": torch.mean(torch.log(out_deg_seq)).item(),
        "var_log_out_deg": torch.var(torch.log(out_deg_seq)).item(),
        "mean_log_in_str": (
            torch.mean(torch.log(in_str_seq) + torch.log(monetary_unit))
        ).item(),
        "var_log_in_str": torch.var(torch.log(in_str_seq)).item(),
        "mean_log_out_str": (
            torch.mean(torch.log(out_str_seq) + torch.log(monetary_unit))
        ).item(),
        "var_log_out_str": torch.var(torch.log(out_str_seq)).item(),
    }


def compute_correlations(
    in_deg_seq: torch.Tensor,
    out_deg_seq: torch.Tensor,
    in_str_seq: torch.Tensor,
    out_str_seq: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute correlation coefficients between network properties.

    Args:
        in_deg_seq: In-degree sequence.
        out_deg_seq: Out-degree sequence.
        in_str_seq: In-strength sequence.
        out_str_seq: Out-strength sequence.

    Returns:
        Dictionary of correlation coefficients as tensors (preserves gradients).
    """
    ki_ko_corr = torch.corrcoef(torch.stack((in_deg_seq.log(), out_deg_seq.log())))[
        0, 1
    ]

    si_so_corr = torch.corrcoef(torch.stack((in_str_seq.log(), out_str_seq.log())))[
        0, 1
    ]

    so_ko_corr = torch.corrcoef(torch.stack((out_str_seq.log(), out_deg_seq.log())))[
        0, 1
    ]

    si_ki_corr = torch.corrcoef(torch.stack((in_str_seq.log(), in_deg_seq.log())))[0, 1]

    return {
        "ki_ko_corr": ki_ko_corr,
        "si_so_corr": si_so_corr,
        "so_ko_corr": so_ko_corr,
        "si_ki_corr": si_ki_corr,
    }


def compute_OLS_regression_coefficients(
    in_deg_seq: torch.Tensor,
    out_deg_seq: torch.Tensor,
    in_str_seq: torch.Tensor,
    out_str_seq: torch.Tensor,
) -> dict[str, float]:
    """Compute regression coefficients between log strength and log degrees, where all 4 variables with observations equal to zero are dropped."""
    si_ki_OLSregcoeff = compute_ols_beta(x=in_deg_seq.log(), y=in_str_seq.log())
    ki_si_OLSregcoeff = compute_ols_beta(x=in_str_seq.log(), y=in_deg_seq.log())
    r2in = si_ki_OLSregcoeff * ki_si_OLSregcoeff
    so_ko_OLSregcoeff = compute_ols_beta(x=out_deg_seq.log(), y=out_str_seq.log())
    ko_so_OLSregcoeff = compute_ols_beta(x=out_str_seq.log(), y=out_deg_seq.log())
    r2out = so_ko_OLSregcoeff * ko_so_OLSregcoeff
    return {
        "si_ki_OLSregcoeff": si_ki_OLSregcoeff,
        "ki_si_OLSregcoeff": ki_si_OLSregcoeff,
        "r2in": r2in,
        "so_ko_OLSregcoeff": so_ko_OLSregcoeff,
        "ko_so_OLSregcoeff": ko_so_OLSregcoeff,
        "r2out": r2out,
    }


def compute_TLS_regression_coefficients(
    in_deg_seq: torch.Tensor,
    out_deg_seq: torch.Tensor,
    in_str_seq: torch.Tensor,
    out_str_seq: torch.Tensor,
) -> dict[str, float]:
    """Compute regression coefficients between log strength and log degrees, where all 4 variables with observations equal to zero are dropped."""
    si_so_TLSregcoeff = compute_tls_slope(x=out_str_seq.log(), y=in_str_seq.log())
    ki_ko_TLSregcoeff = compute_tls_slope(x=out_deg_seq.log(), y=in_deg_seq.log())
    return {
        "si_so_TLSregcoeff": si_so_TLSregcoeff,
        "ki_ko_TLSregcoeff": ki_ko_TLSregcoeff,
    }


def compute_agg_model_percentiles(
    agg_model: torch.Tensor,
    data_iot_actual: torch.Tensor,
) -> tuple[dict[str, float], dict[str, float]]:
    agg_model_flattened = agg_model.flatten()
    data_iot_actual_flattened = data_iot_actual.flatten()
    q25_agg_model, q50_agg_model, q75_agg_model = torch.quantile(
        agg_model_flattened, torch.tensor([0.25, 0.50, 0.75])
    )
    q25_data_iot_actual, q50_data_iot_actual, q75_data_iot_actual = torch.quantile(
        data_iot_actual_flattened, torch.tensor([0.25, 0.50, 0.75])
    )
    summary_agg_model = {
        "rmse": torch.sqrt(criterion_mse(agg_model, data_iot_actual)).item(),
        "mean": agg_model_flattened.mean().item(),
        "min": agg_model_flattened.min().item(),
        "p25": q25_agg_model.item(),
        "median": q50_agg_model.item(),
        "p75": q75_agg_model.item(),
        "max": agg_model_flattened.max().item(),
    }
    summary_data_iot_actual = {
        "mean": data_iot_actual_flattened.mean().item(),
        "min": data_iot_actual_flattened.min().item(),
        "p25": q25_data_iot_actual.item(),
        "median": q50_data_iot_actual.item(),
        "p75": q75_data_iot_actual.item(),
        "max": data_iot_actual_flattened.max().item(),
    }
    return summary_agg_model, summary_data_iot_actual
