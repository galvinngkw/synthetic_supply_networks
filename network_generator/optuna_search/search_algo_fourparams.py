"""Single run of Optuna search algorithm for best four theta parameters, and visualize the optimization process."""

import optuna
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from optuna.visualization.matplotlib import (
    plot_contour,
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_slice,
)
from network_generator.config import NetworkConfig, TargetConfig
from network_generator.initialization import populate_weights_fourparams
from network_generator.metrics import (
    compute_correlations,
    compute_degrees_from_model,
    compute_hill_exponent,
    compute_mean_and_variance_of_log_distributions,
    compute_OLS_regression_coefficients,
    compute_pagerank,
    compute_strengths_from_model,
    compute_TLS_regression_coefficients,
)
from network_generator.utils import scale_by_iot

class optuna_search_single_experiment:
    """Optimize four theta parameters using Optuna and a local combined_loss."""

    def __init__(
        self,
        in_deg: torch.Tensor,
        out_deg: torch.Tensor,
        in_str: torch.Tensor,
        out_str: torch.Tensor,
        adj_mat: torch.Tensor,
        data_iot: torch.Tensor,
        num_industries: int,
        firm_to_industry: torch.Tensor,
        theta_k_in_upper_bound: float,
        theta_k_in_lower_bound: float,
        theta_k_out_upper_bound: float,
        theta_k_out_lower_bound: float,
        theta_f_in_upper_bound: float,
        theta_f_in_lower_bound: float,
        theta_f_out_upper_bound: float,
        theta_f_out_lower_bound: float,
        n_trials: int = 100,
        timeout: float | None = None,
        sampler_seed: int | None = None,
        network_config: NetworkConfig | None = None,
        target_config: TargetConfig | None = None,
    ):
        self.in_deg = in_deg
        self.out_deg = out_deg
        self.in_str = in_str
        self.out_str = out_str
        self.adj_mat = adj_mat
        self.data_iot = data_iot
        self.num_industries = num_industries
        self.firm_to_industry = firm_to_industry
        self.target_config = (
            target_config if target_config is not None else TargetConfig()
        )
        self.network_config = (
            network_config if network_config is not None else NetworkConfig()
        )

        self.theta_k_in_upper_bound = float(theta_k_in_upper_bound)
        self.theta_k_in_lower_bound = float(theta_k_in_lower_bound)
        self.theta_k_out_upper_bound = float(theta_k_out_upper_bound)
        self.theta_k_out_lower_bound = float(theta_k_out_lower_bound)
        self.theta_f_in_upper_bound = float(theta_f_in_upper_bound)
        self.theta_f_in_lower_bound = float(theta_f_in_lower_bound)
        self.theta_f_out_upper_bound = float(theta_f_out_upper_bound)
        self.theta_f_out_lower_bound = float(theta_f_out_lower_bound)
        self.n_trials = int(n_trials)
        self.timeout = timeout
        self.sampler_seed = sampler_seed

        self.study: optuna.study.Study | None = None
        self.best_stats: dict | None = None

    def _compute_statistics_for_params(
        self,
        theta_k_in: float,
        theta_k_out: float,
        theta_f_in: float,
        theta_f_out: float,
    ) -> dict:
        """Network statistics and losses; combined_loss is defined only here for Optuna."""
        wgt_mat_preinit = populate_weights_fourparams(
            self.in_deg,
            self.out_deg,
            self.in_str,
            self.out_str,
            self.adj_mat,
            theta_k_in=theta_k_in,
            theta_k_out=theta_k_out,
            theta_f_in=theta_f_in,
            theta_f_out=theta_f_out,
        )
        wgt_mat_init = scale_by_iot(
            wgt_mat_preinit,
            self.data_iot,
            self.firm_to_industry,
        )

        in_deg, out_deg = compute_degrees_from_model(wgt_mat_init)
        _, edge_non_neg_weights, in_str, out_str = compute_strengths_from_model(
            wgt_mat_init
        )

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
        pagerank = compute_pagerank(wgt_mat_init)
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
            monetary_unit=self.network_config.monetary_unit,
        )

        var_log_in_str = torch.tensor(
            mean_and_variances["var_log_in_str"], dtype=torch.float32
        )
        var_log_out_str = torch.tensor(
            mean_and_variances["var_log_out_str"], dtype=torch.float32
        )

        hill_in_deg = compute_hill_exponent(
            in_deg,
            self.network_config.threshold_percentile_for_hill_deg,
            is_degree=True,
        )
        hill_out_deg = compute_hill_exponent(
            out_deg,
            self.network_config.threshold_percentile_for_hill_deg,
            is_degree=True,
        )
        hill_in_str = compute_hill_exponent(
            in_str[in_str > 0],
            self.network_config.threshold_percentile_for_hill_strength,
        )
        hill_out_str = compute_hill_exponent(
            out_str[out_str > 0],
            self.network_config.threshold_percentile_for_hill_strength,
        )
        hill_influence = compute_hill_exponent(
            pagerank,
            self.network_config.threshold_percentile_for_hill_influence,
        )
        hill_weights = compute_hill_exponent(
            edge_pos_weights,
            self.network_config.threshold_percentile_for_hill_weight,
        )

        def to_tensor(x, dtype=torch.float32):
            if isinstance(x, torch.Tensor):
                return x.detach().to(dtype=dtype)
            return torch.tensor(x, dtype=dtype)

        criterion_mse = nn.MSELoss()

        so_ko_ols = to_tensor(OLS_regression_coefficients["so_ko_OLSregcoeff"])
        ko_so_ols = to_tensor(OLS_regression_coefficients["ko_so_OLSregcoeff"])
        si_ki_ols = to_tensor(OLS_regression_coefficients["si_ki_OLSregcoeff"])
        ki_si_ols = to_tensor(OLS_regression_coefficients["ki_si_OLSregcoeff"])
        si_so_tls = to_tensor(TLS_regression_coefficients["si_so_TLSregcoeff"])

        so_ko_corr_tensor = to_tensor(correlations["so_ko_corr"])
        si_ki_corr_tensor = to_tensor(correlations["si_ki_corr"])

        target_so_ko_regcoeff = torch.tensor(
            self.target_config.target_so_ko_regcoeff, dtype=torch.float32
        )
        target_ko_so_regcoeff = torch.tensor(
            self.target_config.target_ko_so_regcoeff, dtype=torch.float32
        )
        target_si_ki_regcoeff = torch.tensor(
            self.target_config.target_si_ki_regcoeff, dtype=torch.float32
        )
        target_ki_si_regcoeff = torch.tensor(
            self.target_config.target_ki_si_regcoeff, dtype=torch.float32
        )
        target_tls_str = torch.tensor(
            self.target_config.target_tls_str, dtype=torch.float32
        )
        target_so_ko_corr = torch.tensor(
            self.target_config.target_so_ko_corr, dtype=torch.float32
        )
        target_si_ki_corr = torch.tensor(
            self.target_config.target_si_ki_corr, dtype=torch.float32
        )
        target_var_si = torch.tensor(
            self.target_config.target_variance_si, dtype=torch.float32
        )
        target_var_so = torch.tensor(
            self.target_config.target_variance_so, dtype=torch.float32
        )
        target_hill_in_str = torch.tensor(
            self.target_config.target_hill_in_str, dtype=torch.float32
        )
        target_hill_out_str = torch.tensor(
            self.target_config.target_hill_out_str, dtype=torch.float32
        )

        hill_in_str_t = to_tensor(hill_in_str)
        hill_out_str_t = to_tensor(hill_out_str)

        hill_in_str_loss = criterion_mse(hill_in_str_t, target_hill_in_str)
        hill_out_str_loss = criterion_mse(hill_out_str_t, target_hill_out_str)

        var_loss = criterion_mse(var_log_in_str, target_var_si) + criterion_mse(
            var_log_out_str, target_var_so
        )

        reg_coeff_loss = (
            criterion_mse(so_ko_ols, target_so_ko_regcoeff)
            + criterion_mse(ko_so_ols, target_ko_so_regcoeff)
            + criterion_mse(si_ki_ols, target_si_ki_regcoeff)
            + criterion_mse(ki_si_ols, target_ki_si_regcoeff)
        )

        tls_str_loss = criterion_mse(si_so_tls, target_tls_str)

        log_str_deg_corr_loss = criterion_mse(
            so_ko_corr_tensor, target_so_ko_corr
        ) + criterion_mse(si_ki_corr_tensor, target_si_ki_corr)

        combined_loss = (
            hill_in_str_loss
            + hill_out_str_loss
            + log_str_deg_corr_loss
            + reg_coeff_loss
            + tls_str_loss
            + 1e-3 * var_loss
        )

        stats_data = {
            "theta_k_in": theta_k_in,
            "theta_k_out": theta_k_out,
            "theta_f_in": theta_f_in,
            "theta_f_out": theta_f_out,
            "Mean log in-degree": mean_and_variances["mean_log_in_deg"],
            "Var log in-degree": mean_and_variances["var_log_in_deg"],
            "Mean log out-degree": mean_and_variances["mean_log_out_deg"],
            "Var log out-degree": mean_and_variances["var_log_out_deg"],
            "Mean log in-strength": mean_and_variances["mean_log_in_str"],
            "Var log in-strength": mean_and_variances["var_log_in_str"],
            "Mean log out-strength": mean_and_variances["mean_log_out_str"],
            "Var log out-strength": mean_and_variances["var_log_out_str"],
            "Hill in-degree": hill_in_deg.item(),
            "Hill out-degree": hill_out_deg.item(),
            "Hill in-strength": hill_in_str.item(),
            "Hill out-strength": hill_out_str.item(),
            "Hill influence": hill_influence.item(),
            "Hill weights": hill_weights.item(),
            "Corr: ki-ko": correlations["ki_ko_corr"].item(),
            "Corr: si-so": correlations["si_so_corr"].item(),
            "Corr: so-ko": correlations["so_ko_corr"].item(),
            "Corr: si-ki": correlations["si_ki_corr"].item(),
            "OLS: so-ko": OLS_regression_coefficients["so_ko_OLSregcoeff"].item(),
            "OLS: ko-so": OLS_regression_coefficients["ko_so_OLSregcoeff"].item(),
            "OLS: r2out": OLS_regression_coefficients["r2out"].item(),
            "OLS: si-ki": OLS_regression_coefficients["si_ki_OLSregcoeff"].item(),
            "OLS: ki-si": OLS_regression_coefficients["ki_si_OLSregcoeff"].item(),
            "OLS: r2in": OLS_regression_coefficients["r2in"].item(),
            "TLS: si-so": TLS_regression_coefficients["si_so_TLSregcoeff"].item(),
            "TLS: ki-ko": TLS_regression_coefficients["ki_ko_TLSregcoeff"].item(),
            "hill_in_str_loss": hill_in_str_loss.item(),
            "hill_out_str_loss": hill_out_str_loss.item(),
            "log_str_deg_corr_loss": log_str_deg_corr_loss.item(),
            "reg_coeff_loss": reg_coeff_loss.item(),
            "tls_str_loss": tls_str_loss.item(),
            "var_loss": var_loss.item(),
            "combined_loss": combined_loss.item(),
            "prop_of_reg_coeff_loss": (reg_coeff_loss / combined_loss).item(),
        }

        return stats_data

    def _objective(self, trial: optuna.trial.Trial) -> float:
        theta_k_in = trial.suggest_float(
            "theta_k_in", self.theta_k_in_lower_bound, self.theta_k_in_upper_bound
        )
        theta_k_out = trial.suggest_float(
            "theta_k_out", self.theta_k_out_lower_bound, self.theta_k_out_upper_bound
        )
        theta_f_in = trial.suggest_float(
            "theta_f_in", self.theta_f_in_lower_bound, self.theta_f_in_upper_bound
        )
        theta_f_out = trial.suggest_float(
            "theta_f_out", self.theta_f_out_lower_bound, self.theta_f_out_upper_bound
        )

        try:
            stats_data = self._compute_statistics_for_params(
                theta_k_in=theta_k_in,
                theta_k_out=theta_k_out,
                theta_f_in=theta_f_in,
                theta_f_out=theta_f_out,
            )
            trial.set_user_attr("stats_data", stats_data)
            return float(stats_data["combined_loss"])
        except Exception:
            return float("inf")

    def optimize(self) -> dict:
        sampler = optuna.samplers.TPESampler(seed=self.sampler_seed)
        self.study = optuna.create_study(direction="minimize", sampler=sampler)
        self.study.optimize(
            self._objective, n_trials=self.n_trials, timeout=self.timeout
        )

        best_trial = self.study.best_trial
        best_stats = best_trial.user_attrs.get("stats_data")
        if best_stats is None:
            best_stats = self._compute_statistics_for_params(
                theta_k_in=best_trial.params["theta_k_in"],
                theta_k_out=best_trial.params["theta_k_out"],
                theta_f_in=best_trial.params["theta_f_in"],
                theta_f_out=best_trial.params["theta_f_out"],
            )

        self.best_stats = best_stats
        return best_stats

    def get_trials_dataframe(self) -> pd.DataFrame:
        if self.study is None:
            raise ValueError("No study available. Call optimize() first.")

        trial_rows = []
        for trial in self.study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue

            stats_data = trial.user_attrs.get("stats_data")
            if stats_data is None:
                continue

            row = {
                "trial_number": trial.number,
                "theta_k_in": trial.params["theta_k_in"],
                "theta_k_out": trial.params["theta_k_out"],
                "theta_f_in": trial.params["theta_f_in"],
                "theta_f_out": trial.params["theta_f_out"],
                "combined_loss": trial.value,
            }
            row.update(stats_data)
            trial_rows.append(row)

        if not trial_rows:
            return pd.DataFrame()
        return pd.DataFrame(trial_rows)

    def get_best_params(self) -> dict:
        if self.study is None:
            raise ValueError("No study available. Call optimize() first.")
        return dict(self.study.best_params)

    def plot_figures(self, show: bool = False) -> dict:
        """Create Optuna diagnostic visualizations for the current study.

        Args:
            show: If True, call ``fig.show()`` for each figure.

        Returns:
            Dictionary mapping plot names to matplotlib figure objects.
        """
        if self.study is None:
            raise ValueError("No study available. Call optimize() first.")

        # Optuna's matplotlib visualizations can modify global rcParams (style/grid).
        # Keep those changes local so downstream notebook plots are unaffected.
        with mpl.rc_context(rc=mpl.rcParamsDefault):
            mpl.rcParams["axes.grid"] = False
            figures = {
                "optimization_history": plot_optimization_history(self.study),
                "parallel_coordinate": plot_parallel_coordinate(self.study),
                "slice": plot_slice(self.study),
                "contour": plot_contour(self.study),
                "param_importances": plot_param_importances(self.study),
            }
        optimization_history_axes = figures["optimization_history"]
        if isinstance(optimization_history_axes, np.ndarray):
            history_axes = [
                ax
                for ax in optimization_history_axes.ravel()
                if hasattr(ax, "set_yscale")
            ]
        else:
            history_axes = (
                [optimization_history_axes]
                if hasattr(optimization_history_axes, "set_yscale")
                else []
            )
        for ax in history_axes:
            ax.set_yscale("log")
            
        slice_axes = figures["slice"]
        if isinstance(slice_axes, np.ndarray):
            slice_axes = [
                ax 
                for ax in slice_axes.ravel() 
                if hasattr(ax, "set_yscale")
            ]
        else:
            slice_axes = (
                [slice_axes] 
                if hasattr(slice_axes, "set_yscale") 
                else []
                )
        for ax in slice_axes:
            ax.set_yscale("log")

        # Collect unique Figure instances (one Optuna plot may expose multiple Axes).
        seen_fig_ids: set[int] = set()
        unique_figures: list = []
        for ax_obj in figures.values():
            if isinstance(ax_obj, np.ndarray):
                axes_iter = [ax for ax in ax_obj.ravel() if hasattr(ax, "figure")]
            else:
                axes_iter = [ax_obj] if hasattr(ax_obj, "figure") else []
            for ax in axes_iter:
                fig = ax.figure
                fid = id(fig)
                if fid not in seen_fig_ids:
                    seen_fig_ids.add(fid)
                    unique_figures.append(fig)

        if show:
            # Figure.show() uses the canvas's show(), which fails on Agg ("non-interactive").
            # Jupyter inline hooks plt.show(), not per-figure .show(); use pyplot once here.
            plt.show()
        else:
            for fig in unique_figures:
                plt.close(fig)

        return figures
