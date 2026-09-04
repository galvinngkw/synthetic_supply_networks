"""
For Overleaf tables and figures.
Generate multiple networks, compute their network and distributional properties, and output them in a DataFrame
"""

import gc

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from network_generator.config import NetworkConfig, TargetConfig
from network_generator.data_iot import (
    load_input_output_table,
    assign_firms_to_industries,
)
from network_generator.diagnostics import compute_network_stats
from network_generator.initialization import (
    directed_configuration_model,
    generate_burr_degrees,
    generate_burr_degrees_matching_sum,
    generate_initial_strength_fitnesses,
    get_parameters_for_burr_distribution,
    populate_weights_fourparams,
    triangular_wedge_matching,
)
from network_generator.metrics import (
    compute_degrees_from_model,
    compute_hill_exponent,
)

from network_generator.utils import scale_by_iot

# Min variance of log positive degrees; only (num_firms, target_avg_conn) pairs listed here are resampled.
_OUT_DEG_RESAMPLE_MIN_VAR_LOG: dict[tuple[int, int], float] = {
    (10000, 20): 2.3,
    (10000, 40): 2.95,
    (50000, 40): 2.95,
    (100000, 40): 2.95,
    (500000, 40): 2.95,
}
_IN_DEG_RESAMPLE_MIN_VAR_LOG: dict[tuple[int, int], float] = {
    (10000, 20): 1.7,
    (10000, 40): 1.85,
    (50000, 40): 1.95,
    (100000, 40): 1.95,
    (500000, 40): 1.95,
}


def _out_deg_needs_resample(
    seq: torch.Tensor,
    num_firms: int,
    min_log_var: float,
    network_config: NetworkConfig,
) -> bool:
    # Guard against invalid draws before any downstream statistics.
    if seq.numel() == 0 or not torch.isfinite(seq).all():
        return True
    if (seq < 0).any():
        return True
    # For simple directed graphs without self-loops, each node degree is at most N-1.
    if seq.max().item() >= num_firms:
        return True

    seq_pos = seq[seq > 0]
    if seq_pos.numel() == 0:
        return True

    hill_val = compute_hill_exponent(
        seq,
        network_config.threshold_percentile_for_hill_deg,
        is_degree=True,
    ).item()
    log_var = seq_pos.log().var().item()
    mean_deg = seq.float().mean().item()

    if num_firms == 10000:
        return (
            seq.max().item() < 0.1 * num_firms
            or log_var < min_log_var
            or hill_val < 1.0
            or hill_val > 3.0
            or mean_deg < 40.0
        )
    else:
        return (
            seq.max().item() < 0.1 * num_firms
            or log_var < min_log_var
            or hill_val < 1.4
            or hill_val > 1.5
            or mean_deg < 40.0
        )


def _in_deg_needs_resample(
    seq: torch.Tensor, num_firms: int, min_log_var: float, network_config: NetworkConfig
) -> bool:
    # Guard against invalid draws before any downstream statistics.
    if seq.numel() == 0 or not torch.isfinite(seq).all():
        return True
    if (seq < 0).any():
        return True
    # For simple directed graphs without self-loops, each node degree is at most N-1.
    if seq.max().item() >= num_firms:
        return True

    seq_pos = seq[seq > 0]
    if seq_pos.numel() == 0:
        return True

    hill_val = compute_hill_exponent(
        seq,
        network_config.threshold_percentile_for_hill_deg,
        is_degree=True,
    ).item()
    log_var = seq_pos.log().var().item()

    if num_firms == 10000:
        return log_var < min_log_var or hill_val < 2.0 or hill_val > 7.0
    else:
        return log_var < min_log_var or hill_val < 2.3 or hill_val > 2.4


def generate_multiple_networks_and_save_them(
    base_path: str,
    num_experiments: int,
    max_out_deg_resamples: int,
    max_in_deg_resamples: int,
    theta_k_in_value: float,
    theta_k_out_value: float,
    theta_f_in_value: float,
    theta_f_out_value: float,
    network_config: NetworkConfig,
    target_config: TargetConfig,
):
    num_firms = int(network_config.num_firms)
    target_average_connection = torch.tensor(
        target_config.target_average_connection, dtype=torch.float32
    )
    burr_max_indeg = torch.tensor(network_config.burr_max_indeg, dtype=torch.float32)
    burr_max_outdeg = torch.tensor(network_config.burr_max_outdeg, dtype=torch.float32)
    burr_alpha_outdeg = network_config.burr_alpha_outdeg
    burr_var_log_outdeg = network_config.burr_var_log_outdeg
    burr_alpha_indeg = network_config.burr_alpha_indeg
    burr_var_log_indeg = network_config.burr_var_log_indeg

    dict_diagnostics_wgt_mat_init = {}
    itr = 0
    data_iot_torch, _, num_industries = load_input_output_table(
        data_path="data/",
        country_code=network_config.country_code,
        year=network_config.country_year,
        monetary_unit=network_config.monetary_unit,
    )
    print(
        "Please create 2 subfolders in the base_path: wgt_mat_init, firm_to_industry if you haven't."
    )
    with tqdm(total=num_experiments, desc="experiments", unit="exp") as pbar:
        while itr < num_experiments:
            k_outdeg, c_outdeg, lam_outdeg = get_parameters_for_burr_distribution(
                burr_alpha_outdeg, burr_var_log_outdeg, target_average_connection
            )
            k_indeg, c_indeg, lam_indeg = get_parameters_for_burr_distribution(
                burr_alpha_indeg, burr_var_log_indeg, target_average_connection
            )

            try:
                out_deg_seq_burr = generate_burr_degrees(
                    c_val=c_outdeg,
                    d_val=k_outdeg,
                    lam_val=lam_outdeg,
                    max_cutoff=burr_max_outdeg,
                    num_of_firms=num_firms,
                )
                key = (num_firms, int(target_average_connection))
                min_var_log_out = _OUT_DEG_RESAMPLE_MIN_VAR_LOG.get(key)
                if min_var_log_out is not None:
                    out_deg_retry_count = 0
                    while _out_deg_needs_resample(
                        out_deg_seq_burr, num_firms, min_var_log_out, network_config
                    ):
                        out_deg_retry_count += 1
                        if out_deg_retry_count > max_out_deg_resamples:
                            raise RuntimeError(
                                f"out_deg_seq_burr resampling exceeded {max_out_deg_resamples} attempts"
                            )
                        out_deg_seq_burr = generate_burr_degrees(
                            c_val=c_outdeg,
                            d_val=k_outdeg,
                            lam_val=lam_outdeg,
                            max_cutoff=burr_max_outdeg,
                            num_of_firms=num_firms,
                        )

                in_deg_seq_burr = generate_burr_degrees_matching_sum(
                    c_val=c_indeg,
                    d_val=k_indeg,
                    lam_val=lam_indeg,
                    max_cutoff=burr_max_indeg,
                    num_of_firms=num_firms,
                    degree_sum=out_deg_seq_burr.sum(),
                )
                min_var_log_in = _IN_DEG_RESAMPLE_MIN_VAR_LOG.get(key)
                if min_var_log_in is not None:
                    in_deg_retry_count = 0
                    while _in_deg_needs_resample(
                        in_deg_seq_burr, num_firms, min_var_log_in, network_config
                    ):
                        in_deg_retry_count += 1
                        if in_deg_retry_count > max_in_deg_resamples:
                            raise RuntimeError(
                                f"in_deg_seq_burr resampling exceeded {max_in_deg_resamples} attempts"
                            )
                        in_deg_seq_burr = generate_burr_degrees_matching_sum(
                            c_val=c_indeg,
                            d_val=k_indeg,
                            lam_val=lam_indeg,
                            max_cutoff=burr_max_indeg,
                            num_of_firms=num_firms,
                            degree_sum=out_deg_seq_burr.sum(),
                        )
            except RuntimeError as e:
                tqdm.write(
                    f"[itr={itr}] Restarting experiment due to guarded resampling loop: {e}"
                )
                gc.collect()
                continue

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
                monetary_unit=network_config.monetary_unit,
            )

            # re-running assign_firms_to_industries will generate a different firm_to_industry_tensor. it is a random process.
            firm_to_industry_tensor = assign_firms_to_industries(
                num_firms, data_iot_torch
            )

            torch.save(
                firm_to_industry_tensor,
                f"{base_path}/firm_to_industry/firm_to_industry_tensor_v{itr}.pth",
            )

            wgt_mat_pre_init = populate_weights_fourparams(
                in_deg_seq_init,
                out_deg_seq_init,
                in_str_seq_init,
                out_str_seq_init,
                adj_mat_init,
                theta_k_in=theta_k_in_value,
                theta_k_out=theta_k_out_value,
                theta_f_in=theta_f_in_value,
                theta_f_out=theta_f_out_value,
            )

            wgt_mat_init = scale_by_iot(
                wgt_mat_pre_init, data_iot_torch, firm_to_industry_tensor
            )

            torch.save(
                wgt_mat_init, f"{base_path}/wgt_mat_init/wgt_mat_init_v{itr}.pth"
            )

            df_diagnostics_wgt_mat_optim_init = compute_network_stats(
                model=wgt_mat_init,
                firm_to_industry=firm_to_industry_tensor,
                data_iot=data_iot_torch,
                num_of_industries=num_industries,
                network_config=network_config,
                target_config=target_config,
            )

            # compute_network_stats formats some counts with thousands separators;
            # strip commas before coercion so they are not parsed as NaN.
            df_diagnostics_wgt_mat_optim_init.Value = pd.to_numeric(
                df_diagnostics_wgt_mat_optim_init.Value.astype(str).str.replace(
                    ",", "", regex=False
                ),
                errors="coerce",
            )
            for metric, value in zip(
                df_diagnostics_wgt_mat_optim_init.Metric,
                df_diagnostics_wgt_mat_optim_init.Value.to_numpy(),
            ):
                dict_diagnostics_wgt_mat_init.setdefault(metric, []).append(value)

            del (
                k_outdeg,
                c_outdeg,
                lam_outdeg,
                k_indeg,
                c_indeg,
                lam_indeg,
                out_deg_seq_burr,
                in_deg_seq_burr,
                test_reordered_out_deg_seq_burr,
                adj_mat_init,
                in_deg_seq_init,
                out_deg_seq_init,
                in_str_seq_init,
                out_str_seq_init,
                firm_to_industry_tensor,
                wgt_mat_init,
            )
            gc.collect()

            itr += 1
            pbar.update(1)

    dict_diagnostics_wgt_mat_init = {
        k: np.array(v) for k, v in dict_diagnostics_wgt_mat_init.items()
    }
    df_compiled_diagnostics_wgt_mat_init = pd.DataFrame.from_dict(
        dict_diagnostics_wgt_mat_init, orient="index"
    )
    df_compiled_diagnostics_wgt_mat_init["mean"] = (
        df_compiled_diagnostics_wgt_mat_init.mean(axis=1)
    )
    df_compiled_diagnostics_wgt_mat_init["std"] = (
        df_compiled_diagnostics_wgt_mat_init.std(axis=1)
    )
    df_compiled_diagnostics_wgt_mat_init.to_csv(
        f"{base_path}/wgt_mat_init/{num_experiments}_experiments_compiled_diagnostics_for_wgt_mat_init.csv"
    )
    return df_compiled_diagnostics_wgt_mat_init
