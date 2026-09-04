"""Convert diagnostics DataFrames to LaTeX tables for Overleaf figures."""

import pandas as pd

TARGET_LABEL_MAP = {
    "Number of nodes": "B",
    "Number of edges": "B",
    "Share of supplier-only firms": "NT",
    "Share of customer-only firms": "NT",
    "Mean degree": "B",
    "Max in-degree": "B",
    "Max out-degree": "B",
    "Mean log in-degree": "NT",
    "Var log in-degree": "B",
    "Mean log out-degree": "NT",
    "Var log out-degree": "B",
    "Mean log in-strength (1USD)": "NT",
    "Var log in-strength": "W",
    "Mean log out-strength (1USD)": "NT",
    "Var log out-strength": "W",
    "Giant component size": "NT",
    "Avg path length": "NT",
    "Degree assortativity": "NT",
    "Reciprocity": "NT",
    "Avg clustering": "NT",
    "Global clustering": "NT",
    "Hill in-degree": "B",
    "Hill out-degree": "B",
    "Hill in-strength": "W",
    "Hill out-strength": "W",
    "Hill influence": "NT",
    "Hill weights": "NT",
    "Corr: ki-ko": "B",
    "Corr: si-so": "NT",
    "Corr: so-ko": "W",
    "Corr: si-ki": "W",
    "OLS: so-ko": "W",
    "OLS: ko-so": "W",
    "OLS: si-ki": "W",
    "OLS: ki-si": "W",
    "TLS: si-so": "W",
    "TLS: ki-ko": "NT",
}

ADDLINESPACE_BEFORE_LEFT = {
    "Max in-degree",
    "Mean log in-strength (1USD)",
    "Giant component size",
}

ADDLINESPACE_BEFORE_RIGHT = {
    "Corr: ki-ko",
    "OLS: so-ko",
    "TLS: si-so",
    "IOT RMSE",
}


def _target_label(metric: str) -> str:
    if metric.startswith("IOT RMSE"):
        return "W"
    return TARGET_LABEL_MAP.get(metric, "NT")


def _needs_addlinespace(metric: str, addlinespace_before: set[str]) -> bool:
    if metric in addlinespace_before:
        return True
    return metric.startswith("IOT RMSE") and "IOT RMSE" in addlinespace_before


def _format_cell(val) -> str:
    if val is None:
        return ""
    if isinstance(val, (list, tuple)):
        return "" if len(val) == 0 else str(list(val))
    if hasattr(val, "tolist") and not isinstance(val, (str, bytes)):
        val = val.tolist()
        if isinstance(val, list):
            return "" if len(val) == 0 else str(val)
    try:
        if pd.isna(val):
            return ""
    except (ValueError, TypeError):
        pass
    s = str(val).strip()
    return "" if s.lower() in {"nan", "none", ""} else s


def _panel_to_latex(
    panel: pd.DataFrame,
    addlinespace_before: set[str],
) -> str:
    lines = [
        r"\begin{tabular}[t]{@{}lrrc@{}}",
        r"Metric & Synthetic & Empirical & Target \\",
        r"\cmidrule(lr){1-4}",
    ]

    for _, row in panel.iterrows():
        metric = row["Metric_original"]
        if _needs_addlinespace(metric, addlinespace_before):
            lines.append(r"\cmidrule(lr){1-4}")

        lines.append(
            f"{row['Metric']} & {_format_cell(row['Synthetic'])} & "
            f"{_format_cell(row['Empirical'])} & {row['Target']} \\\\"
        )

    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def diagnostics_to_latextable(
    df: pd.DataFrame,
    split_after_metric: str = "Global clustering",
) -> str:
    """Format a diagnostics DataFrame as a two-panel LaTeX table."""
    df = df.copy()
    df["Metric_original"] = df["Metric"]

    rename_map = {
        "Share of supplier-only firms": r"Share of $k^{\mathrm{in}}=0$",
        "Share of customer-only firms": r"Share of $k^{\mathrm{out}}=0$",
        "Max in-degree": r"Max $k^{\mathrm{in}}$",
        "Max out-degree": r"Max $k^{\mathrm{out}}$",
        "Mean log in-degree": r"Mean log $k^{\mathrm{in}}$",
        "Var log in-degree": r"Var log $k^{\mathrm{in}}$",
        "Mean log out-degree": r"Mean log $k^{\mathrm{out}}$",
        "Var log out-degree": r"Var log $k^{\mathrm{out}}$",
        "Mean log in-strength (1USD)": r"Mean log $s^{\mathrm{in}}$",
        "Var log in-strength": r"Var log $s^{\mathrm{in}}$",
        "Mean log out-strength (1USD)": r"Mean log $s^{\mathrm{out}}$",
        "Var log out-strength": r"Var log $s^{\mathrm{out}}$",
        "Giant component size": "LWCC",
        "Hill in-degree": r"Hill $\alpha$, $k^{\mathrm{in}}$",
        "Hill out-degree": r"Hill $\alpha$, $k^{\mathrm{out}}$",
        "Hill in-strength": r"Hill $\alpha$, $s^{\mathrm{in}}$",
        "Hill out-strength": r"Hill $\alpha$, $s^{\mathrm{out}}$",
        "Hill influence": r"Hill $\alpha$, influence",
        "Hill weights": r"Hill $\alpha$, weights",
        "Corr: ki-ko": r"Corr: $k^{\mathrm{in}}\sim k^{\mathrm{out}}$",
        "Corr: si-so": r"Corr: $s^{\mathrm{in}}\sim s^{\mathrm{out}}$",
        "Corr: so-ko": r"Corr: $s^{\mathrm{out}}\sim k^{\mathrm{out}}$",
        "Corr: si-ki": r"Corr: $s^{\mathrm{in}}\sim k^{\mathrm{in}}$",
        "OLS: so-ko": r"OLS: $s^{\mathrm{out}}\sim k^{\mathrm{out}}$",
        "OLS: ko-so": r"OLS: $k^{\mathrm{out}}\sim s^{\mathrm{out}}$",
        "OLS: r2out": r"$R^2$(out)",
        "OLS: si-ki": r"OLS: $s^{\mathrm{in}}\sim k^{\mathrm{in}}$",
        "OLS: ki-si": r"OLS: $k^{\mathrm{in}}\sim s^{\mathrm{in}}$",
        "OLS: r2in": r"$R^2$(in)",
        "TLS: si-so": r"TLS: $s^{\mathrm{in}}\sim s^{\mathrm{out}}$",
        "TLS: ki-ko": r"TLS: $k^{\mathrm{in}}\sim k^{\mathrm{out}}$",
    }

    df["Metric"] = df["Metric"].replace(rename_map)

    agg_rows_to_drop = [
        m
        for m in df["Metric_original"]
        if m.startswith("IOT mean")
        or m.startswith("IOT min")
        or m.startswith("IOT p25")
        or m.startswith("IOT median")
        or m.startswith("IOT p75")
        or m.startswith("IOT max")
        or m.startswith("OLS: r2")
    ]
    df = df[~df["Metric_original"].isin(agg_rows_to_drop)].reset_index(drop=True)
    df = df.rename(columns={"Value": "Synthetic"})
    df["Target"] = df["Metric_original"].map(_target_label)

    split_key = split_after_metric
    if split_key not in df["Metric"].values:
        split_key = "Global clustering"

    split_idx = df.index[df["Metric"].eq(split_key)][0] + 1
    left = df.iloc[:split_idx].reset_index(drop=True)

    right_rest = df.iloc[split_idx:].reset_index(drop=True)
    iot_rmse_mask = right_rest["Metric_original"].str.startswith("IOT RMSE")
    right = pd.concat(
        [right_rest[~iot_rmse_mask], right_rest[iot_rmse_mask]],
        ignore_index=True,
    )

    left_latex = _panel_to_latex(left, ADDLINESPACE_BEFORE_LEFT)
    right_latex = _panel_to_latex(right, ADDLINESPACE_BEFORE_RIGHT)

    return "\n".join(
        [
            r"\begin{threeparttable}",
            "",
            r"\begin{tabular}{@{}c@{\hspace{0.04\textwidth}}c@{}}",
            r"\toprule",
            "",
            left_latex,
            "",
            "&",
            "",
            right_latex,
            "",
            r"\\",
            r"\bottomrule",
            r"\end{tabular}",
            "",
            r"\begin{tablenotes}",
            r"\footnotesize",
            r"\item[B] Targeted when creating adjacency matrix $A_{ij}$.",
            r"\item[W] Targeted when creating weights $W_{ij}$ on existing links $A_{ij}$.",
            r"\item[NT] Not targeted.",
            r"\end{tablenotes}",
            r"\end{threeparttable}",
        ]
    )
