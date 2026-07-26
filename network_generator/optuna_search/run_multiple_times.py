"""Wrapper script to run multiple Optuna experiments to find the average theta parameters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch    
from tqdm import tqdm

from network_generator.config import NetworkConfig, TargetConfig
from network_generator.data_iot import load_input_output_table, assign_firms_to_industries
from network_generator.initialization import (
    directed_configuration_model,
    generate_burr_degrees,
    generate_burr_degrees_matching_sum,
    get_parameters_for_burr_distribution,
    generate_initial_strength_fitnesses,
    triangular_wedge_matching,
)
from network_generator.metrics import compute_degrees_from_model, compute_hill_exponent
from network_generator.optuna_search.search_algo_fourparams import optuna_search_single_experiment

THETA_KEYS = (
    "theta_k_in",
    "theta_k_out",
    "theta_f_in",
    "theta_f_out",
)


def _sample_out_degrees(
    *,
    c_outdeg: float,
    k_outdeg: float,
    lam_outdeg: float,
    max_outdeg: torch.Tensor,
    num_firms: int,
    network_config: NetworkConfig,
) -> torch.Tensor:
    out_deg_seq_burr = generate_burr_degrees(
        c_val=c_outdeg,
        d_val=k_outdeg,
        lam_val=lam_outdeg,
        max_cutoff=max_outdeg,
        num_of_firms=num_firms,
    )
    while (
        (out_deg_seq_burr.max().item() < 0.1 * num_firms)
        or (out_deg_seq_burr[out_deg_seq_burr > 0].log().var().item() < 2.95)
        or (
            compute_hill_exponent(
                out_deg_seq_burr,
                network_config.threshold_percentile_for_hill_deg,
                is_degree=True,
            ).item()
            > 1.5
        )
        or (
            compute_hill_exponent(
                out_deg_seq_burr,
                network_config.threshold_percentile_for_hill_deg,
                is_degree=True,
            ).item()
            < 1.4
        )
        or (out_deg_seq_burr.float().mean() < 40.0)
    ):
        out_deg_seq_burr = generate_burr_degrees(
            c_val=c_outdeg,
            d_val=k_outdeg,
            lam_val=lam_outdeg,
            max_cutoff=max_outdeg,
            num_of_firms=num_firms,
        )
    return out_deg_seq_burr


def _sample_in_degrees(
    *,
    c_indeg: float,
    k_indeg: float,
    lam_indeg: float,
    max_indeg: torch.Tensor,
    num_firms: int,
    out_deg_seq_burr: torch.Tensor,
    network_config: NetworkConfig,
) -> torch.Tensor:
    in_deg_seq_burr = generate_burr_degrees_matching_sum(
        c_val=c_indeg,
        d_val=k_indeg,
        lam_val=lam_indeg,
        max_cutoff=max_indeg,
        num_of_firms=num_firms,
        degree_sum=out_deg_seq_burr.sum(),
    )
    while (
        (in_deg_seq_burr.min().item() < 0)
        or (in_deg_seq_burr[in_deg_seq_burr > 0].log().var().item() < 1.95)
        or (
            compute_hill_exponent(
                in_deg_seq_burr,
                network_config.threshold_percentile_for_hill_deg,
                is_degree=True,
            ).item()
            < 2.3
        )
        or (
            compute_hill_exponent(
                in_deg_seq_burr,
                network_config.threshold_percentile_for_hill_deg,
                is_degree=True,
            ).item()
            > 2.4
        )
    ):
        in_deg_seq_burr = generate_burr_degrees_matching_sum(
            c_val=c_indeg,
            d_val=k_indeg,
            lam_val=lam_indeg,
            max_cutoff=max_indeg,
            num_of_firms=num_firms,
            degree_sum=out_deg_seq_burr.sum(),
        )
    return in_deg_seq_burr


def run_optuna_theta_multiple_experiments(
    *,
    num_experiments: int,
    network_config: NetworkConfig,
    target_config: TargetConfig,
    data_path: str | Path = "data/",
    n_trials: int = 200,
    sampler_seed: int | None = None,
    theta_k_in_upper_bound: float = 2.0,
    theta_k_in_lower_bound: float = -2.0,
    theta_k_out_upper_bound: float = 2.0,
    theta_k_out_lower_bound: float = -2.0,
    theta_f_in_upper_bound: float = 2.0,
    theta_f_in_lower_bound: float = -2.0,
    theta_f_out_upper_bound: float = 2.0,
    theta_f_out_lower_bound: float = -2.0,
) -> pd.DataFrame:
    """For each experiment, redraw degrees/strengths/graph, run Optuna, record best thetas.

    Returns a DataFrame with index ``THETA_KEYS`` (4 rows), ``num_experiments`` columns
    named ``exp_0`` ... ``exp_{n-1}``, plus ``mean`` and ``std`` (sample std, ``ddof=1``)
    computed across the experiment columns only.
    """
    if num_experiments < 1:
        raise ValueError("num_experiments must be at least 1")

    num_firms = int(network_config.num_firms)
    monetary_unit = float(network_config.monetary_unit)
    burr_alpha_outdeg = float(network_config.burr_alpha_outdeg)
    burr_var_log_outdeg = float(network_config.burr_var_log_outdeg)
    burr_alpha_indeg = float(network_config.burr_alpha_indeg)
    burr_var_log_indeg = float(network_config.burr_var_log_indeg)
    target_average_connection = float(target_config.target_average_connection)
    burr_max_outdeg = torch.tensor(
        network_config.burr_max_outdeg, dtype=torch.float32
    )
    burr_max_indeg = torch.tensor(
        network_config.burr_max_indeg, dtype=torch.float32
    )

    data_iot_torch, _data_iot_np, num_industries = load_input_output_table(
        data_path=data_path,
        country_code=network_config.country_code,
        year=network_config.country_year,
        monetary_unit=monetary_unit,
    )

    rows: list[list[float]] = []
    rep_range: Any = range(num_experiments)

    for _ in tqdm(rep_range, desc="Optuna experiments", unit="exp"):

        k_outdeg, c_outdeg, lam_outdeg = get_parameters_for_burr_distribution(
            burr_alpha_outdeg, burr_var_log_outdeg, target_average_connection
        )
        k_indeg, c_indeg, lam_indeg = get_parameters_for_burr_distribution(
            burr_alpha_indeg, burr_var_log_indeg, target_average_connection
        )

        out_deg_seq_burr = _sample_out_degrees(
            c_outdeg=c_outdeg,
            k_outdeg=k_outdeg,
            lam_outdeg=lam_outdeg,
            max_outdeg=burr_max_outdeg,
            num_firms=num_firms,
            network_config=network_config,
        )
        in_deg_seq_burr = _sample_in_degrees(
            c_indeg=c_indeg,
            k_indeg=k_indeg,
            lam_indeg=lam_indeg,
            max_indeg=burr_max_indeg,
            num_firms=num_firms,
            out_deg_seq_burr=out_deg_seq_burr,
            network_config=network_config,
        )

        test_reordered_out_deg_seq_burr = triangular_wedge_matching(
            in_deg_seq=in_deg_seq_burr,
            out_deg_seq=out_deg_seq_burr,
            noise_strength=network_config.noise_amount_for_reordering_burrkiko,
        )

        adj_mat_init = directed_configuration_model(
            in_deg_seq_burr, test_reordered_out_deg_seq_burr
        )
        in_deg_seq_init, out_deg_seq_init = compute_degrees_from_model(adj_mat_init)

        in_str_seq_init, out_str_seq_init = generate_initial_strength_fitnesses(
            mean_in_str=network_config.mean_for_gen_lognorm_fi,
            mean_out_str=network_config.mean_for_gen_lognorm_fo,
            var_in_str=in_deg_seq_init[in_deg_seq_init > 0].log().var().item(),
            var_out_str=out_deg_seq_init[out_deg_seq_init > 0].log().var().item(),
            correlation=network_config.corr_lognorm_gen_fi_fo,
            num_of_firms=num_firms,
            monetary_unit=monetary_unit,
        )

        firm_to_industry_tensor = assign_firms_to_industries(num_firms, data_iot_torch)

        optuna_runner = optuna_search_single_experiment(
            in_deg=in_deg_seq_init,
            out_deg=out_deg_seq_init,
            in_str=in_str_seq_init,
            out_str=out_str_seq_init,
            adj_mat=adj_mat_init,
            data_iot=data_iot_torch,
            num_industries=num_industries,
            firm_to_industry=firm_to_industry_tensor,
            theta_k_in_lower_bound=theta_k_in_lower_bound,
            theta_k_in_upper_bound=theta_k_in_upper_bound,
            theta_k_out_lower_bound=theta_k_out_lower_bound,
            theta_k_out_upper_bound=theta_k_out_upper_bound,
            theta_f_in_lower_bound=theta_f_in_lower_bound,
            theta_f_in_upper_bound=theta_f_in_upper_bound,
            theta_f_out_lower_bound=theta_f_out_lower_bound,
            theta_f_out_upper_bound=theta_f_out_upper_bound,
            n_trials=n_trials,
            sampler_seed=sampler_seed,
            target_config=target_config,
            network_config=network_config,
        )
        run_stats = optuna_runner.optimize()
        rows.append([float(run_stats[k]) for k in THETA_KEYS])

    exp_cols = [f"exp_{i}" for i in range(num_experiments)]
    base = pd.DataFrame(rows, columns=list(THETA_KEYS)).T
    base.columns = exp_cols
    out = base.copy()
    out["mean"] = base.mean(axis=1)
    out["std"] = base.std(axis=1, ddof=1)
    return out
