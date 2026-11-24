"""End-to-end: from a country and year to a synthetic firm-level network consistent with its input-output table.

    from network_generator.pipeline import generate_synthetic_network

    result = generate_synthetic_network("HUN", 2015, num_firms=100_000, seed=1)
    result.iot_report()      # industry aggregates vs the OECD target
    result.stats()           # network statistics vs the empirical targets
    result.save("out/hun_2015")

The steps are the ones in ``runner.ipynb``:

1. get the target table (local ``data/``, cache or mirror; see ``network_generator.oecd_icio``);
2. draw Burr in- and out-degree sequences, redrawing until they pass ``DegreeAcceptance``;
3. reorder out-degrees against in-degrees (triangular wedge) and realise the topology with the directed configuration model;
4. draw lognormal strength fitnesses;
5. optionally tune the four weight exponents with Optuna (``optimize=True``), otherwise use the values in ``NetworkConfig``;
6. populate weights and rescale them so the industry aggregate matches the target table.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from network_generator.config import DATA_PATH, NetworkConfig, TargetConfig, TrainingConfig
from network_generator.data_iot import assign_firms_to_industries, load_input_output_table
from network_generator.initialization import (
    directed_configuration_model,
    generate_burr_degrees,
    generate_burr_degrees_matching_sum,
    generate_initial_strength_fitnesses,
    get_parameters_for_burr_distribution,
    populate_weights_fourparams,
    triangular_wedge_matching,
)
from network_generator.metrics import compute_degrees_from_model, compute_hill_exponent
from network_generator.oecd_icio.store import DEFAULT_RELEASE
from network_generator.utils import firm_model_to_iot, scale_by_iot


@dataclass(frozen=True)
class DegreeAcceptance:
    """Bounds a Burr degree draw must satisfy before it is used.

    Defaults are the ones used for the paper (``runner.ipynb``). ``None`` disables a bound. The max-degree bound is a fraction of the number of firms.
    """

    out_min_max_degree_fraction: float | None = 0.1
    out_min_log_var: float | None = 2.95
    out_hill_range: tuple[float, float] | None = (1.4, 1.5)
    out_min_mean: float | None = 40.0
    in_min_log_var: float | None = 1.95
    in_hill_range: tuple[float, float] | None = (2.3, 2.4)
    max_attempts: int = 10_000

    @classmethod
    def none(cls) -> "DegreeAcceptance":
        """Accept the first draw (useful for small test networks)."""
        return cls(None, None, None, None, None, None)


def _log_var(seq: torch.Tensor) -> float:
    return seq[seq > 0].float().log().var().item()


def _hill(seq: torch.Tensor, percentile: float) -> float:
    return compute_hill_exponent(seq, percentile, is_degree=True).item()


def _in_range(value: float, bounds: tuple[float, float] | None) -> bool:
    return bounds is None or bounds[0] <= value <= bounds[1]


def _valid_degrees(seq: torch.Tensor, n: int) -> bool:
    """Draws no simple directed graph can realise, rejected whatever the acceptance bounds (as in the paper scripts)."""
    return (
        seq.numel() > 0
        and bool(torch.isfinite(seq.float()).all())
        and seq.min().item() >= 0
        and seq.max().item() < n  # at most n - 1 neighbours without self-loops
        and bool((seq > 0).any())
    )


def _accept_out(seq, n, rule: DegreeAcceptance, pct: float) -> bool:
    if not _valid_degrees(seq, n):
        return False
    return (
        (rule.out_min_max_degree_fraction is None or seq.max().item() >= rule.out_min_max_degree_fraction * n)
        and (rule.out_min_log_var is None or _log_var(seq) >= rule.out_min_log_var)
        and (rule.out_min_mean is None or seq.float().mean().item() >= rule.out_min_mean)
        and _in_range(_hill(seq, pct), rule.out_hill_range)
    )


def _accept_in(seq, n, rule: DegreeAcceptance, pct: float) -> bool:
    if not _valid_degrees(seq, n):
        return False
    return (
        (rule.in_min_log_var is None or _log_var(seq) >= rule.in_min_log_var)
        and _in_range(_hill(seq, pct), rule.in_hill_range)
    )


def _draw_until(draw, accept, what: str, max_attempts: int) -> tuple[torch.Tensor, int]:
    for attempt in range(1, max_attempts + 1):
        seq = draw()
        if accept(seq):
            return seq, attempt
    raise RuntimeError(
        f"no {what} sequence passed DegreeAcceptance in {max_attempts} draws; "
        "relax the bounds (e.g. for small num_firms use DegreeAcceptance.none())"
    )


def seed_everything(seed: int) -> None:
    """Seed every RNG the pipeline touches: Python (igraph, firm assignment), NumPy (scipy's Burr sampling) and torch.

    These are the process-wide generators, so seeding also affects any other code in the process that uses them.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class SyntheticNetwork:
    weights: torch.Tensor
    """Sparse COO firm x firm weight matrix, in ``network_config.monetary_unit``; entry (i, j) is what firm j buys from firm i."""
    firm_to_industry: torch.Tensor
    target_iot: torch.Tensor
    """Industry x industry target, in ``network_config.monetary_unit``."""
    network_config: NetworkConfig
    target_config: TargetConfig
    thetas: dict[str, float]
    release: str
    seed: int | None
    draws: dict[str, int] = field(default_factory=dict)
    optuna_trials: pd.DataFrame | None = None

    @property
    def num_industries(self) -> int:
        return self.target_iot.shape[0]

    def aggregate_iot(self) -> torch.Tensor:
        """The synthetic network summed back to industry x industry."""
        return firm_model_to_iot(self.weights, self.firm_to_industry, self.num_industries)

    def iot_report(self) -> pd.Series:
        """How well the industry aggregate reproduces the target table.

        ``scale_by_iot`` can only rescale industry pairs that have at least one firm-to-firm edge, so target flows between pairs without edges are lost; ``uncovered_*`` measures that loss.
        """
        target = self.target_iot.double()
        agg = self.aggregate_iot().double()
        uncovered = (target > 0) & (agg == 0)
        return pd.Series(
            {
                "target_total": target.sum().item(),
                "synthetic_total": agg.sum().item(),
                "max_abs_cell_error": (agg - target).abs().max().item(),
                "relative_total_error": ((agg.sum() - target.sum()) / target.sum()).item(),
                "uncovered_cells": int(uncovered.sum().item()),
                "uncovered_share_of_flow": (target[uncovered].sum() / target.sum()).item(),
                "num_edges": int(self.weights._nnz()),
                "num_firms": self.network_config.num_firms,
            }
        )

    def stats(self) -> pd.DataFrame:
        """Network statistics against the empirical targets (``diagnostics.compute_network_stats``)."""
        from network_generator.diagnostics import compute_network_stats

        return compute_network_stats(
            model=self.weights,
            firm_to_industry=self.firm_to_industry,
            data_iot=self.target_iot,
            num_of_industries=self.num_industries,
            network_config=self.network_config,
            target_config=self.target_config,
        )

    def edge_list(self) -> pd.DataFrame:
        """Edges as a DataFrame: supplier, buyer, weight, supplier/buyer industry index."""
        w = self.weights.coalesce()
        src, dst = w.indices()
        return pd.DataFrame(
            {
                "supplier": src.numpy(),
                "buyer": dst.numpy(),
                "weight": w.values().numpy(),
                "supplier_industry": self.firm_to_industry[src].numpy(),
                "buyer_industry": self.firm_to_industry[dst].numpy(),
            }
        )

    def save(self, out_dir: str | Path, edge_format: str = "csv") -> Path:
        """Write the edge list, ``firms.csv``, ``target_iot.csv`` and ``run.json``.

        ``edge_format`` is ``"csv"`` (default, no extra dependency) or ``"parquet"`` (smaller and faster for large networks; needs ``pyarrow``).
        """
        if edge_format not in ("csv", "parquet"):
            raise ValueError(f"edge_format must be 'csv' or 'parquet', not {edge_format!r}")
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        edges = self.edge_list()
        if edge_format == "parquet":
            edges.to_parquet(out / "edges.parquet", index=False)
        else:
            edges.to_csv(out / "edges.csv", index=False)
        pd.DataFrame({"firm": np.arange(len(self.firm_to_industry)), "industry": self.firm_to_industry.numpy()}).to_csv(out / "firms.csv", index=False)
        pd.DataFrame(self.target_iot.numpy()).to_csv(out / "target_iot.csv", index=False)
        run = {
            "country_code": self.network_config.country_code,
            "year": self.network_config.country_year,
            "release": self.release,
            "seed": self.seed,
            "thetas": self.thetas,
            "draws": self.draws,
            "network_config": self.network_config.model_dump(),
            "iot_report": self.iot_report().to_dict(),
        }
        (out / "run.json").write_text(json.dumps(run, indent=2, default=float) + "\n")
        if self.optuna_trials is not None:
            self.optuna_trials.to_csv(out / "optuna_trials.csv", index=False)
        return out


def generate_synthetic_network(
    country: str | None = None,
    year: int | None = None,
    num_firms: int | None = None,
    *,
    network_config: NetworkConfig | None = None,
    target_config: TargetConfig | None = None,
    training_config: TrainingConfig | None = None,
    optimize: bool = False,
    acceptance: DegreeAcceptance = DegreeAcceptance(),
    seed: int | None = None,
    data_path: str | Path | None = DATA_PATH,
    release: str = DEFAULT_RELEASE,
) -> SyntheticNetwork:
    """Generate a synthetic firm network whose industry aggregate matches ``country``'s ``year`` table.

    ``country``, ``year`` and ``num_firms`` override the corresponding ``network_config`` fields (YAML defaults otherwise). With ``optimize=True`` the four weight exponents are tuned with Optuna under ``training_config``; otherwise the values in ``network_config`` are used.

    ``seed`` seeds the process-wide Python, NumPy and torch generators (see ``seed_everything``). ``data_path`` defaults to the repository's ``data/`` in a checkout and to no local folder when installed; see ``load_input_output_table``.
    """
    overrides = {
        k: v
        for k, v in {"country_code": country, "country_year": year, "num_firms": num_firms}.items()
        if v is not None
    }
    base = network_config or NetworkConfig()
    network_config = base.model_copy(update=overrides) if overrides else base
    target_config = target_config or TargetConfig()
    if seed is not None:
        seed_everything(seed)

    n = network_config.num_firms
    pct = network_config.threshold_percentile_for_hill_deg

    # 1. target table
    iot, _, num_industries = load_input_output_table(
        data_path=data_path,
        country_code=network_config.country_code,
        year=network_config.country_year,
        monetary_unit=network_config.monetary_unit,
        release=release,
    )

    # 2. Burr degrees
    avg_conn = torch.tensor(target_config.target_average_connection, dtype=torch.float32)
    k_out, c_out, lam_out = get_parameters_for_burr_distribution(network_config.burr_alpha_outdeg, network_config.burr_var_log_outdeg, avg_conn)
    k_in, c_in, lam_in = get_parameters_for_burr_distribution(network_config.burr_alpha_indeg, network_config.burr_var_log_indeg, avg_conn)
    max_out = torch.tensor(network_config.burr_max_outdeg, dtype=torch.float32)
    max_in = torch.tensor(network_config.burr_max_indeg, dtype=torch.float32)

    out_deg, out_draws = _draw_until(
        lambda: generate_burr_degrees(c_val=c_out, d_val=k_out, lam_val=lam_out, max_cutoff=max_out, num_of_firms=n),
        lambda s: _accept_out(s, n, acceptance, pct),
        "out-degree",
        acceptance.max_attempts,
    )
    in_deg, in_draws = _draw_until(
        lambda: generate_burr_degrees_matching_sum(c_val=c_in, d_val=k_in, lam_val=lam_in, max_cutoff=max_in, num_of_firms=n, degree_sum=out_deg.sum()),
        lambda s: _accept_in(s, n, acceptance, pct),
        "in-degree",
        acceptance.max_attempts,
    )

    # 3. topology
    out_deg = triangular_wedge_matching(in_deg_seq=in_deg, out_deg_seq=out_deg, noise_strength=network_config.noise_amount_for_reordering_burrkiko)
    adj = directed_configuration_model(in_deg, out_deg)
    in_deg_real, out_deg_real = compute_degrees_from_model(adj)

    # 4. strength fitnesses
    in_str, out_str = generate_initial_strength_fitnesses(
        mean_in_str=network_config.mean_for_gen_lognorm_fi,
        mean_out_str=network_config.mean_for_gen_lognorm_fo,
        var_in_str=_log_var(in_deg_real),
        var_out_str=_log_var(out_deg_real),
        correlation=network_config.corr_lognorm_gen_fi_fo,
        num_of_firms=n,
        monetary_unit=network_config.monetary_unit,
    )
    firm_to_industry = assign_firms_to_industries(n, iot)

    # 5. weight exponents
    thetas = {
        "theta_k_in": network_config.theta_k_in_value,
        "theta_k_out": network_config.theta_k_out_value,
        "theta_f_in": network_config.theta_f_in_value,
        "theta_f_out": network_config.theta_f_out_value,
    }
    trials = None
    if optimize:
        from network_generator.optuna_search.search_algo_fourparams import optuna_search_single_experiment

        training_config = training_config or TrainingConfig()
        runner = optuna_search_single_experiment(
            in_deg=in_deg_real,
            out_deg=out_deg_real,
            in_str=in_str,
            out_str=out_str,
            adj_mat=adj,
            data_iot=iot,
            num_industries=num_industries,
            firm_to_industry=firm_to_industry,
            theta_k_in_upper_bound=training_config.theta_k_in_upper_bound,
            theta_k_in_lower_bound=training_config.theta_k_in_lower_bound,
            theta_k_out_upper_bound=training_config.theta_k_out_upper_bound,
            theta_k_out_lower_bound=training_config.theta_k_out_lower_bound,
            theta_f_in_upper_bound=training_config.theta_f_in_upper_bound,
            theta_f_in_lower_bound=training_config.theta_f_in_lower_bound,
            theta_f_out_upper_bound=training_config.theta_f_out_upper_bound,
            theta_f_out_lower_bound=training_config.theta_f_out_lower_bound,
            n_trials=training_config.n_trials,
            sampler_seed=seed,
            target_config=target_config,
            network_config=network_config,
        )
        runner.optimize()
        thetas = {k: float(v) for k, v in runner.get_best_params().items()}
        trials = runner.get_trials_dataframe()

    # 6. weights, rescaled to the table
    weights = populate_weights_fourparams(
        in_deg_real, out_deg_real, in_str, out_str, adj,
        thetas["theta_k_in"], thetas["theta_k_out"], thetas["theta_f_in"], thetas["theta_f_out"],
    )
    weights = scale_by_iot(weights, iot, firm_to_industry)

    return SyntheticNetwork(
        weights=weights,
        firm_to_industry=firm_to_industry,
        target_iot=iot,
        network_config=network_config,
        target_config=target_config,
        thetas=thetas,
        release=release,
        seed=seed,
        draws={"out_degree": out_draws, "in_degree": in_draws},
        optuna_trials=trials,
    )
