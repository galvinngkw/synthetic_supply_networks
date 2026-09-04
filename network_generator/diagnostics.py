"""Compute network properties for a given firm-level weighted network, output as a DataFrame."""
import random

import igraph as ig
import pandas as pd
import torch
from network_generator.config import NetworkConfig, TargetConfig
from network_generator.metrics import (
    compute_agg_model_percentiles,
    compute_correlations,
    compute_degrees_from_model,
    compute_hill_exponent,
    compute_mean_and_variance_of_log_distributions,
    compute_OLS_regression_coefficients,
    compute_pagerank,
    compute_strengths_from_model,
    compute_supplieronly_customeronly_from_model,
    compute_TLS_regression_coefficients,
)
from network_generator.utils import firm_model_to_iot


# Helper function to extract value from tensor or float
def get_value(x):
    return x.item() if isinstance(x, torch.Tensor) else x


def build_empirical_target_mapping(
    network_config: NetworkConfig,
    target_config: TargetConfig,
    *,
    num_firms: int | None = None,
    summary_data_iot_actual: dict[str, object] | None = None,
) -> dict[str, object]:
    """Map diagnostic metric names to empirical / target reference values.

    Shared by ``compute_network_stats`` (Empirical column) and the robustness-check
    LaTeX table. Values come from ``target_config``, fixed calibration constants,
    and optionally observed IOT summary statistics.
    """
    if num_firms is None:
        num_firms = network_config.num_firms

    target_mapping: dict[str, object] = {
        "Number of nodes": f"{num_firms:,}",
        "Number of edges": f"{int(num_firms * target_config.target_average_connection):,}",
        "Mean degree": target_config.target_average_connection,
        "Mean log in-degree": 3.0,
        "Var log in-degree": 2.0,
        "Mean log out-degree": 2.0,
        "Var log out-degree": 3.0,
        "Mean log in-strength (1USD)": 10.0,
        "Var log in-strength": 9.0,
        "Mean log out-strength (1USD)": 10.0,
        "Var log out-strength": 8.0,
        "Avg path length": 3.0,
        "Degree assortativity": [-0.2, -0.01],
        "Reciprocity": [0.03, 0.05],
        "Avg clustering": [0.19, 0.28],
        "Global clustering": "low",
        "Hill in-degree": target_config.target_hill_in_deg,
        "Hill out-degree": target_config.target_hill_out_deg,
        "Hill in-strength": target_config.target_hill_in_str,
        "Hill out-strength": target_config.target_hill_out_str,
        "Hill influence": [1.2, 1.3],
        "Hill weights": [1.1, 1.2],
        "Corr: ki-ko": target_config.target_ki_ko_corr,
        "Corr: si-so": target_config.target_si_so_corr,
        "Corr: so-ko": target_config.target_so_ko_corr,
        "Corr: si-ki": target_config.target_si_ki_corr,
        "OLS: so-ko": target_config.target_so_ko_regcoeff,
        "OLS: ko-so": target_config.target_ko_so_regcoeff,
        "OLS: r2out": f"{target_config.target_so_ko_corr**2:.2f}",
        "OLS: si-ki": target_config.target_si_ki_regcoeff,
        "OLS: ki-si": target_config.target_ki_si_regcoeff,
        "OLS: r2in": f"{target_config.target_si_ki_corr**2:.2f}",
        "TLS: si-so": target_config.target_tls_str,
        "TLS: ki-ko": target_config.target_tls_deg,
    }

    if summary_data_iot_actual is not None:
        musd = int(network_config.monetary_unit / 1e6)
        target_mapping.update(
            {
                f"IOT mean ({musd} MUSD)": summary_data_iot_actual["mean"],
                f"IOT min ({musd} MUSD)": summary_data_iot_actual["min"],
                f"IOT p25 ({musd} MUSD)": summary_data_iot_actual["p25"],
                f"IOT median ({musd} MUSD)": summary_data_iot_actual["median"],
                f"IOT p75 ({musd} MUSD)": summary_data_iot_actual["p75"],
                f"IOT max ({musd} MUSD)": summary_data_iot_actual["max"],
            }
        )

    return target_mapping


def compute_network_stats(
    model: torch.Tensor,
    firm_to_industry: torch.Tensor,
    data_iot: torch.Tensor,
    num_of_industries: int = 45,
    num_samples_for_path_length: int = 500,
    network_config: NetworkConfig | None = None,
    target_config: TargetConfig | None = None,
) -> pd.DataFrame:
    """Compute comprehensive network statistics.

    Args:
        model: Sparse COO tensor representing the network.
        num_samples_for_path_length: Number of samples for path length estimation.

    Returns:
        DataFrame with all computed statistics.
    """
    
    model = model.coalesce()
    row_idx, col_idx = model.indices()
    num_firms = model.size(0)
    num_edges = model._nnz()

    # Aggregate model
    agg_model = firm_model_to_iot(
        model, firm_to_industry=firm_to_industry, num_industries=num_of_industries
    )
    summary_agg_model, summary_data_iot_actual = compute_agg_model_percentiles(
        agg_model, data_iot
    )

    # Build igraph representations
    edges = list(zip(row_idx.numpy(), col_idx.numpy()))

    # Directed graph for reciprocity
    ig_dir = ig.Graph(n=num_firms, edges=edges, directed=True)
    ig_dir.simplify(multiple=True, loops=True)

    ig_undir = ig.Graph(n=num_firms, edges=edges, directed=False)
    ig_undir.simplify(multiple=True, loops=True)

    # Giant component
    components = ig_undir.components(mode="WEAK")
    giant = components.giant()

    # Approximate average path length
    approx_path_length = None
    if giant.vcount() > 0:
        samples = random.sample(
            range(giant.vcount()),
            min(num_samples_for_path_length, giant.vcount()),
        )
        avg_lengths = []
        for v in samples:
            sp = giant.shortest_paths(v, None, mode="ALL")[0]
            finite_sp = [d for d in sp if d < float("inf")]
            if finite_sp:
                avg_lengths.append(sum(finite_sp) / len(finite_sp))
        if avg_lengths:
            approx_path_length = sum(avg_lengths) / len(avg_lengths)

    # Compute network metrics
    in_deg, out_deg = compute_degrees_from_model(model)
    supplier_only_pct, customer_only_pct = compute_supplieronly_customeronly_from_model(
        model
    )
    _, edge_non_neg_weights, in_str, out_str = compute_strengths_from_model(model)

    in_deg_four_positive = in_deg[
        (in_deg > 0) & (out_deg > 0) & (in_str > 0) & (out_str > 0)
    ]
    out_deg_four_positive = out_deg[
        (in_deg > 0) & (out_deg > 0) & (in_str > 0) & (out_str > 0)
    ]
    in_str_four_positive = in_str[
        (in_deg > 0) & (out_deg > 0) & (in_str > 0) & (out_str > 0)
    ]
    out_str_four_positive = out_str[
        (in_deg > 0) & (out_deg > 0) & (in_str > 0) & (out_str > 0)
    ]

    in_deg_deg_positive = in_deg[(in_deg > 0) & (out_deg > 0)]
    out_deg_deg_positive = out_deg[(in_deg > 0) & (out_deg > 0)]
    in_str_str_positive = in_str[(in_str > 0) & (out_str > 0)]
    out_str_str_positive = out_str[(in_str > 0) & (out_str > 0)]

    edge_pos_weights = edge_non_neg_weights[edge_non_neg_weights > 0]
    pagerank = compute_pagerank(model)
    correlations = compute_correlations(
        in_deg_four_positive,
        out_deg_four_positive,
        in_str_four_positive,
        out_str_four_positive,
    )
    OLS_regression_coefficients = compute_OLS_regression_coefficients(
        in_deg_four_positive,
        out_deg_four_positive,
        in_str_four_positive,
        out_str_four_positive,
    )
    TLS_regression_coefficients = compute_TLS_regression_coefficients(
        in_deg_four_positive,
        out_deg_four_positive,
        in_str_four_positive,
        out_str_four_positive,
    )
    mean_and_variances = compute_mean_and_variance_of_log_distributions(
        in_deg_deg_positive,
        out_deg_deg_positive,
        in_str_str_positive,
        out_str_str_positive,
        monetary_unit=network_config.monetary_unit,
    )

    # Compute Hill exponents
    hill_in_deg = compute_hill_exponent(
        in_deg, network_config.threshold_percentile_for_hill_deg, is_degree=True
    )
    hill_out_deg = compute_hill_exponent(
        out_deg, network_config.threshold_percentile_for_hill_deg, is_degree=True
    )
    hill_in_str = compute_hill_exponent(
        in_str, network_config.threshold_percentile_for_hill_strength
    )
    hill_out_str = compute_hill_exponent(
        out_str, network_config.threshold_percentile_for_hill_strength
    )
    hill_influence = compute_hill_exponent(
        pagerank, network_config.threshold_percentile_for_hill_influence
    )
    hill_weights = compute_hill_exponent(
        edge_pos_weights, network_config.threshold_percentile_for_hill_weight
    )

    stats_data = {
        "Metric": [
            "Number of nodes",
            "Number of edges",
            "Share of supplier-only firms",
            "Share of customer-only firms",
            "Mean degree",
            "Max in-degree",
            "Max out-degree",
            "Mean log in-degree",
            "Var log in-degree",
            "Mean log out-degree",
            "Var log out-degree",
            "Mean log in-strength (1USD)",
            "Var log in-strength",
            "Mean log out-strength (1USD)",
            "Var log out-strength",
            "Giant component size",
            "Avg path length",
            "Degree assortativity",
            "Reciprocity",
            "Avg clustering",
            "Global clustering",
            f"IOT RMSE ({int(network_config.monetary_unit/1e6)} MUSD)",
            f"IOT mean ({int(network_config.monetary_unit/1e6)} MUSD)",
            f"IOT min ({int(network_config.monetary_unit/1e6)} MUSD)",
            f"IOT p25 ({int(network_config.monetary_unit/1e6)} MUSD)",
            f"IOT median ({int(network_config.monetary_unit/1e6)} MUSD)",
            f"IOT p75 ({int(network_config.monetary_unit/1e6)} MUSD)",
            f"IOT max ({int(network_config.monetary_unit/1e6)} MUSD)",
            "Hill in-degree",
            "Hill out-degree",
            "Hill in-strength",
            "Hill out-strength",
            "Hill influence",
            "Hill weights",
            "Corr: ki-ko",
            "Corr: si-so",
            "Corr: so-ko",
            "Corr: si-ki",
            "OLS: so-ko",
            "OLS: ko-so",
            "OLS: r2out",
            "OLS: si-ki",
            "OLS: ki-si",
            "OLS: r2in",
            "TLS: si-so",
            "TLS: ki-ko",
        ],
        "Value": [
            f"{num_firms:,}",
            f"{int(num_edges):,}",
            f"{supplier_only_pct:.3g}",
            f"{customer_only_pct:.3g}",
            f"{in_deg.mean():.3g}",
            f"{in_deg.max():,.0f}",
            f"{out_deg.max():,.0f}",
            f"{mean_and_variances['mean_log_in_deg']:.3g}",
            f"{mean_and_variances['var_log_in_deg']:.3g}",
            f"{mean_and_variances['mean_log_out_deg']:.3g}",
            f"{mean_and_variances['var_log_out_deg']:.3g}",
            f"{mean_and_variances['mean_log_in_str']:.3g}",
            f"{mean_and_variances['var_log_in_str']:.3g}",
            f"{mean_and_variances['mean_log_out_str']:.3g}",
            f"{mean_and_variances['var_log_out_str']:.3g}",
            f"{giant.vcount():,}",
            f"{approx_path_length:.3g}" if approx_path_length else "N/A",
            f"{ig_dir.assortativity_degree(directed=True):.3g}",
            f"{ig_dir.reciprocity():.3g}",
            f"{giant.transitivity_avglocal_undirected(mode='zero'):.3g}",
            f"{giant.transitivity_undirected():.3g}",
            f"{summary_agg_model['rmse']:.2e}",
            summary_agg_model["mean"],
            summary_agg_model["min"],
            summary_agg_model["p25"],
            summary_agg_model["median"],
            summary_agg_model["p75"],
            summary_agg_model["max"],
            f"{hill_in_deg.item():.3g}",
            f"{hill_out_deg.item():.3g}",
            f"{hill_in_str.item():.3g}",
            f"{hill_out_str.item():.3g}",
            f"{hill_influence.item():.3g}",
            f"{hill_weights.item():.3g}",
            f"{correlations['ki_ko_corr']:.3g}",
            f"{correlations['si_so_corr']:.3g}",
            f"{correlations['so_ko_corr']:.3g}",
            f"{correlations['si_ki_corr']:.3g}",
            f"{get_value(OLS_regression_coefficients['so_ko_OLSregcoeff']):.3g}",
            f"{get_value(OLS_regression_coefficients['ko_so_OLSregcoeff']):.3g}",
            f"{get_value(OLS_regression_coefficients['r2out']):.3g}",
            f"{get_value(OLS_regression_coefficients['si_ki_OLSregcoeff']):.3g}",
            f"{get_value(OLS_regression_coefficients['ki_si_OLSregcoeff']):.3g}",
            f"{get_value(OLS_regression_coefficients['r2in']):.3g}",
            f"{get_value(TLS_regression_coefficients['si_so_TLSregcoeff']):.3g}",
            f"{get_value(TLS_regression_coefficients['ki_ko_TLSregcoeff']):.3g}",
        ],
    }

    stats_df = pd.DataFrame(stats_data)

    target_mapping = build_empirical_target_mapping(
        network_config,
        target_config,
        num_firms=num_firms,
        summary_data_iot_actual=summary_data_iot_actual,
    )
    stats_df["Empirical"] = stats_df["Metric"].map(target_mapping).fillna("")

    return stats_df
