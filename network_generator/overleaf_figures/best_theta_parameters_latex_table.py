import pandas as pd

def dataframe_to_best_theta_parameters_latextable(
    df: pd.DataFrame,
    exclude_last_n: int = 2,
    sigfigs: int = 3
) -> str:
    """
    Convert a dataframe of optimal theta parameters to a LaTeX summary table.

    Assumes df has parameter names as the index:
    theta_k_in, theta_k_out, theta_f_in, theta_f_out

    For theta_f_in and theta_f_out, negative and positive values are summarized separately.
    """

    def get_values(row_name: str) -> pd.Series:
        values = df.loc[row_name]

        if exclude_last_n > 0:
            values = values.iloc[:-exclude_last_n]

        return pd.to_numeric(values, errors="coerce").dropna()

    def fmt(x: float) -> str:
        if pd.isna(x):
            return ""
        return f"${x:.{sigfigs}g}$"

    def fmt_pair(x1: float, x2: float) -> str:
        return f"{fmt(x1)}, {fmt(x2)}"

    def split_mean_std(row_name: str) -> tuple[str, str]:
        values = get_values(row_name)

        negative_values = values[values < 0]
        positive_values = values[values > 0]

        mean_cell = fmt_pair(
            negative_values.mean(),
            positive_values.mean()
        )

        std_cell = fmt_pair(
            negative_values.std(),
            positive_values.std()
        )

        return mean_cell, std_cell

    theta_k_in = get_values("theta_k_in")
    theta_k_out = get_values("theta_k_out")

    theta_f_in_mean, theta_f_in_std = split_mean_std("theta_f_in")
    theta_f_out_mean, theta_f_out_std = split_mean_std("theta_f_out")

    summary_df = pd.DataFrame({
        "Parameter": [
            r"$\theta_{k,\mathrm{in}}$",
            r"$\theta_{k,\mathrm{out}}$",
            r"$\theta_{f,\mathrm{in}}$",
            r"$\theta_{f,\mathrm{out}}$",
        ],
        "Mean": [
            fmt(theta_k_in.mean()),
            fmt(theta_k_out.mean()),
            theta_f_in_mean,
            theta_f_out_mean,
        ],
        r"\multicolumn{1}{l}{Standard deviation}": [
            fmt(theta_k_in.std()),
            fmt(theta_k_out.std()),
            theta_f_in_std,
            theta_f_out_std,
        ],
    })

    return summary_df.to_latex(
        index=False,
        escape=False,
        column_format=r"@{}lcc@{}",
    )