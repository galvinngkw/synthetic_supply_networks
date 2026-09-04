"""Convert robustness-check diagnostics DataFrames to a LaTeX table for Overleaf."""

import pandas as pd

from network_generator.config import NetworkConfig, TargetConfig
from network_generator.diagnostics import build_empirical_target_mapping
from .diagnostics_latex_table import _format_cell

METRIC_RENAME_MAP = {
    "Max in-degree": r"Max $k^{\text{in}}$",
    "Max out-degree": r"Max $k^{\text{out}}$",
    "Mean log in-degree": r"Mean log $k^{\text{in}}$",
    "Var log in-degree": r"Var log $k^{\text{in}}$",
    "Mean log out-degree": r"Mean log $k^{\text{out}}$",
    "Var log out-degree": r"Var log $k^{\text{out}}$",
    "Mean log in-strength (1USD)": r"Mean log $s^{\text{in}}$",
    "Var log in-strength": r"Var log $s^{\text{in}}$",
    "Mean log out-strength (1USD)": r"Mean log $s^{\text{out}}$",
    "Var log out-strength": r"Var log $s^{\text{out}}$",
    "Giant component size": "LWCC",
    "Hill in-degree": r"Hill $\alpha$, $k^{\text{in}}$",
    "Hill out-degree": r"Hill $\alpha$, $k^{\text{out}}$",
    "Hill in-strength": r"Hill $\alpha$, $s^{\text{in}}$",
    "Hill out-strength": r"Hill $\alpha$, $s^{\text{out}}$",
    "Hill influence": r"Hill $\alpha$, influence",
    "Hill weights": r"Hill $\alpha$, weights",
    "Corr: ki-ko": r"Corr: $k^{\text{in}}\sim k^{\text{out}}$",
    "Corr: si-so": r"Corr: $s^{\text{in}}\sim s^{\text{out}}$",
    "Corr: so-ko": r"Corr: $s^{\text{out}}\sim k^{\text{out}}$",
    "Corr: si-ki": r"Corr: $s^{\text{in}}\sim k^{\text{in}}$",
    "OLS: so-ko": r"OLS: $s^{\text{out}}\sim k^{\text{out}}$",
    "OLS: ko-so": r"OLS: $k^{\text{out}}\sim s^{\text{out}}$",
    "OLS: r2out": r"$R^2$(out)",
    "OLS: si-ki": r"OLS: $s^{\text{in}}\sim k^{\text{in}}$",
    "OLS: ki-si": r"OLS: $k^{\text{in}}\sim s^{\text{in}}$",
    "OLS: r2in": r"$R^2$(in)",
    "TLS: si-so": r"TLS: $s^{\text{in}}\sim s^{\text{out}}$",
    "TLS: ki-ko": r"TLS: $k^{\text{in}}\sim k^{\text{out}}$",
}

THETA_ROWS = [
    (r"$\theta_{k,\mathrm{in}}$", "theta_k_in_value"),
    (r"$\theta_{k,\mathrm{out}}$", "theta_k_out_value"),
    (r"$\theta_{f,\mathrm{in}}$", "theta_f_in_value"),
    (r"$\theta_{f,\mathrm{out}}$", "theta_f_out_value"),
]

THETA_KEY_ALIASES = {
    "theta_k_in_value": ("theta_k_in_value", "theta_k_in"),
    "theta_k_out_value": ("theta_k_out_value", "theta_k_out"),
    "theta_f_in_value": ("theta_f_in_value", "theta_f_in"),
    "theta_f_out_value": ("theta_f_out_value", "theta_f_out"),
}

METRIC_ORDER = [
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
]

MIDRULE_AFTER_METRICS = {
    "Mean degree",
    "Var log out-degree",
    "Var log out-strength",
    "Global clustering",
    "Hill weights",
    "Corr: si-ki",
    "OLS: r2in",
    "TLS: ki-ko",
}

COMMENTED_OUT_METRICS = {"OLS: r2out", "OLS: r2in"}

INTEGER_METRICS = {
    "Number of nodes",
    "Number of edges",
    "Giant component size",
    "Max in-degree",
    "Max out-degree",
}


def _normalize_metric_name(metric: str) -> str:
    """Align compiled-diagnostics index names with ``compute_network_stats`` names."""
    if metric.startswith("%"):
        return metric[1:]
    if metric.startswith("RMSE ("):
        return metric.replace("RMSE (", "IOT RMSE (", 1)
    return metric


def _metric_label(metric: str) -> str:
    """Convert an internal metric name to its LaTeX row label."""
    if metric.startswith("IOT RMSE ("):
        return metric
    return METRIC_RENAME_MAP.get(metric, metric)


def _format_theta(val, sigfigs: int) -> str:
    """Format a single theta parameter value for the LaTeX table."""
    if pd.isna(val):
        return ""
    return f"{val:.{sigfigs}g}"


def _get_theta(theta_dict: dict[str, float], key: str) -> float | None:
    """Look up a theta value, accepting both ``theta_k_in_value`` and ``theta_k_in`` keys."""
    for alias in THETA_KEY_ALIASES.get(key, (key,)):
        if alias in theta_dict:
            return theta_dict[alias]
    return None


def _format_mean_std(mean, std, metric: str, sigfigs: int) -> str:
    """Format a synthetic diagnostics cell as ``mean (std)`` for LaTeX output."""
    if pd.isna(mean):
        return ""

    if metric in INTEGER_METRICS:
        mean_str = f"{int(round(mean)):,}"
        if pd.isna(std):
            std_str = "0"
        else:
            std_str = f"{int(round(std)):,}"
    else:
        mean_str = f"{mean:.{sigfigs}g}"
        std_str = f"{std:.{sigfigs}g}" if not pd.isna(std) else "0"

    cell = f"{mean_str} ({std_str})"
    if metric == "Degree assortativity" and mean < 0:
        cell = f"${cell}$"
    return cell


def _format_empirical_number(val, sigfigs: int = 3) -> str:
    """Format a single empirical reference value for the LaTeX table."""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return _format_cell(val)
    if isinstance(val, float) and val.is_integer():
        return str(int(val))
    if isinstance(val, int):
        return str(val)
    formatted = f"{val:.{sigfigs}g}"
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted


def _format_empirical(metric: str, val, sigfigs: int = 3) -> str:
    """Format an empirical reference value, including ranges and assortativity bounds."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    if metric == "Degree assortativity" and isinstance(val, (list, tuple)) and len(val) == 2:
        return f"$[{val[0]},{val[1]}] < 0$"
    if isinstance(val, (list, tuple)):
        if len(val) == 0:
            return ""
        return "[" + ", ".join(_format_empirical_number(x, sigfigs) for x in val) + "]"
    if isinstance(val, str) and "," in val:
        return val
    return _format_empirical_number(val, sigfigs)


def _row_from_metric(
    metric: str,
    df_pos: pd.DataFrame,
    df_neg: pd.DataFrame,
    df_rounded_neg: pd.DataFrame,
    empirical_lookup: dict[str, object],
    sigfigs: int,
) -> str:
    """Build one LaTeX table row for a diagnostic metric across all four columns."""
    label = _metric_label(metric)
    pos_cell = _format_mean_std(
        df_pos.loc[metric, "mean"], df_pos.loc[metric, "std"], metric, sigfigs
    )
    neg_cell = _format_mean_std(
        df_neg.loc[metric, "mean"], df_neg.loc[metric, "std"], metric, sigfigs
    )
    rounded_cell = _format_mean_std(
        df_rounded_neg.loc[metric, "mean"],
        df_rounded_neg.loc[metric, "std"],
        metric,
        sigfigs,
    )
    empirical_cell = _format_empirical(
        metric, empirical_lookup.get(metric, ""), sigfigs
    )

    line = (
        f"{label} & {pos_cell} & {neg_cell} & {rounded_cell} & {empirical_cell} \\\\"
    )
    if metric in COMMENTED_OUT_METRICS:
        return f"% {line}"
    return line


def dataframe_to_robustnesscheck_pos_neg_roundedneg_latextable(
    df_pos: pd.DataFrame,
    df_neg: pd.DataFrame,
    df_rounded_neg: pd.DataFrame,
    network_config: NetworkConfig,
    target_config: TargetConfig,
    theta_pos: dict[str, float] | None = None,
    theta_neg: dict[str, float] | None = None,
    theta_rounded_neg: dict[str, float] | None = None,
    sigfigs: int = 3,
) -> str:
    """
    Merge robustness-check diagnostics from three dataframes and return LaTeX tabular code.

    Each input diagnostics dataframe must contain columns named 'mean' and 'std'.
    The index is assumed to contain the diagnostic/statistic names.

    Empirical values are derived from ``network_config`` and ``target_config``, using the
    same mapping as ``compute_network_stats``. Metric labels follow the same renaming
    rules as ``diagnostics_to_latextable``.

    Each ``theta_*`` dict should map parameter names to values, e.g.
    ``{"theta_k_in_value": -0.0265, "theta_k_out_value": -0.697, ...}``.
    The shorter names ``theta_k_in``, ``theta_k_out``, etc. are also accepted.
    """
    theta_pos = theta_pos or {}
    theta_neg = theta_neg or {}
    theta_rounded_neg = theta_rounded_neg or {}

    def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
        prepared = df[["mean", "std"]].copy()
        prepared.index = [_normalize_metric_name(name) for name in prepared.index]
        return prepared

    df_pos = _prepare_df(df_pos)
    df_neg = _prepare_df(df_neg)
    df_rounded_neg = _prepare_df(df_rounded_neg)
    empirical_lookup = build_empirical_target_mapping(network_config, target_config)

    lines = [
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        "Property & Positive & Negative & Rounded negative & Empirical \\\\",
        r"\midrule",
    ]

    for label, key in THETA_ROWS:
        lines.append(
            f"{label} & "
            f"{_format_theta(_get_theta(theta_pos, key), sigfigs)} & "
            f"{_format_theta(_get_theta(theta_neg, key), sigfigs)} & "
            f"{_format_theta(_get_theta(theta_rounded_neg, key), sigfigs)} & "
            "\\\\"
        )
    lines.append(r"\midrule")

    available_metrics = set(df_pos.index) & set(df_neg.index) & set(df_rounded_neg.index)
    metrics = [metric for metric in METRIC_ORDER if metric in available_metrics]

    iot_rmse_metrics = sorted(
        metric
        for metric in available_metrics
        if metric.startswith("IOT RMSE (") and metric not in metrics
    )
    metrics.extend(iot_rmse_metrics)

    for metric in metrics:
        lines.append(
            _row_from_metric(
                metric,
                df_pos,
                df_neg,
                df_rounded_neg,
                empirical_lookup,
                sigfigs,
            )
        )
        if metric in MIDRULE_AFTER_METRICS:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)
