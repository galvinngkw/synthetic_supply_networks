# Synthetic Supply Networks

A PyTorch-based generator for firm-level supply networks calibrated to empirical input–output tables and distributional targets (degrees, strengths, correlations, Hill exponents, and related statistics).

## Overview

This repository builds synthetic supply networks by:

- **Heavy-tailed degrees (Burr XII)**: In- and out-degree sequences are drawn from truncated Burr distributions; parameters are chosen so that mean, log-variance, and Hill exponent match configuration targets.
- **Degree reordering**: `triangular_wedge_matching` reorders degrees; `noise_amount_for_reordering_burrkiko` in `conf/network/default.yaml` controls the wedge noise (ζ; default −0.14).
- **Sparse topology**: A directed configuration model (`igraph`) realizes the degree sequences by sampling an adjacency matrix. 
- **Lognormal fitnesses**: Initial in- and out-fitnesses sequences are correlated lognormals. 
- **Four-parameter edge weights**: Weights use `populate_weights_fourparams` with $\theta_{k,\mathrm{in}}, \theta_{k,\mathrm{out}},\theta_{f,\mathrm{in}}, \theta_{f,\mathrm{out}}$ (defaults in YAML; optimizable via Optuna).
- **IOT consistency**: `scale_by_iot` rescales the firm–firm network so its industry-level aggregate matches the country’s input–output table (`data_iot.load_input_output_table`).
- **Optuna tuning**: `TrainingConfig` drives Optuna search over the four θ bounds and trial count (`conf/training/default.yaml`). The single-run API is `optuna_search_single_experiment` in `network_generator/optuna_search/search_algo_fourparams.py`; batch replication uses `run_optuna_theta_multiple_experiments` in `run_multiple_times.py`.

**Metrics and diagnostics**: `network_generator/metrics.py` implements degrees, strengths, Hill exponents, OLS/TLS regressions, PageRank, correlations, and related quantities. `diagnostics.compute_network_stats` returns a wide property table for a weighted network.

## Requirements

### Python packages

Install (versions depend on your environment; a typical set is):

```bash
pip install torch numpy pandas matplotlib pydantic pyyaml scipy python-igraph optuna tqdm
```

- **torch**: sparse tensors, tensors throughout the pipeline
- **scipy**: Burr parameter solving (`optimize.root_scalar`, `stats.burr12`)
- **python-igraph**: directed configuration model
- **optuna**: θ hyperparameter search (TPE sampler)
- **tqdm**: progress bars in batch Optuna runs and Overleaf batch scripts

### Data

Place OECD inter-country input–output tables as pickles under `data/`, named `{COUNTRY_CODE}.pkl` (e.g. `HUN.pkl`). Each file is expected to map year → industry × industry matrix in millions USD; the loader rescales by `monetary_unit` in `NetworkConfig` (see `data_iot.load_input_output_table`).

Default `conf/network/default.yaml` uses Hungary (`HUN`) and year `2015`.

## Installation

1. Clone the repository and enter the project root:

   ```bash
   git clone <repository-url>
   cd synthetic_supply_networks
   ```

2. Install dependencies (see above).

3. Ensure `data/{country_code}.pkl` exists for the country and years you use in `NetworkConfig`.

## Configuration

YAML defaults live under `conf/` and are merged by Pydantic models in `network_generator/config.py`:

| File | Role |
|------|------|
| `conf/network/default.yaml` | Firms count, country/year, monetary unit, Burr targets and cutoffs, reordering noise ζ, lognormal strength parameters, default θ values, Hill percentile thresholds |
| `conf/target/default.yaml` | Targets for optimization loss (Hill exponents, moments, correlations, regression slopes, TLS terms) |
| `conf/training/default.yaml` | Optuna: `num_experiments`, `n_trials`, and box constraints for each θ |

Override in Python when constructing configs, for example:

```python
network_config = NetworkConfig(num_firms=100_000, country_code="BEL")
training_config = TrainingConfig(n_trials=300)
```

## Running the project

### Main notebook: `runner.ipynb`

Full interactive workflow. Run cells top to bottom. At a high level:

1. **Imports** – `config`, `data_iot`, `initialization`, `metrics`, `utils`, `visualization`, `diagnostics`, and Optuna search class.
2. **Configs** – `NetworkConfig`, `TargetConfig`, `TrainingConfig`.
3. **Target tensors** – unpack `TargetConfig` for use in losses inside Optuna.
4. **Burr parameters** – `get_parameters_for_burr_distribution` for in- and out-degree sampling.
5. **Out- / in-degree sequences** – generate and resample until validation (Hill, variance, mean, max) passes.
6. **Reorder** – `triangular_wedge_matching` with `noise_amount_for_reordering_burrkiko`.
7. **Topology** – `directed_configuration_model`; recover degrees with `compute_degrees_from_model`.
8. **Strengths** – `generate_initial_strength_fitnesses`.
9. **IOT** – `load_input_output_table` and `assign_firms_to_industries` (assignment is **proportional to industry sales**, with integer rounding adjusted to hit `num_firms`; rerunning changes the firm–industry map).
10. **Optuna** – build `optuna_search_single_experiment`, call `optimize()`, inspect `get_trials_dataframe()` and `plot_figures()`.
11. **Post-process** – a later cell builds a scaled network with `populate_weights_fourparams` and `scale_by_iot` (the notebook currently uses the θ values from `network_config` by default; you can plug in `best_optuna_stats` for the θ fields if desired).
12. **Diagnostics & plots** – `compute_network_stats`, `visualize_network`, degree/strength heatmaps, aggregated IOT vs truth, CCDFs.

Degree resampling loops and Optuna with many trials can be slow for large `num_firms`.

### Batch Optuna: `additional_notebooks/runner_for_optuna_multiple_times.ipynb`

Uses `TrainingConfig` and `run_optuna_theta_multiple_experiments` to repeat the random network draw + Optuna many times and aggregate best θ statistics (see `network_generator/optuna_search/run_multiple_times.py`).

### Other notebooks (under `additional_notebooks/`)

- **`runner_for_systematic_tuning_of_params.ipynb`** – grids over Burr-related controls (e.g. α, ν) for degree statistics.
- **`runner_tuning_zeta_for_reordering_adjmat.ipynb`** – explores the reordering noise parameter ζ.
- **`runner_for_overleaf_maindoc.ipynb`** – builds figures for the main research paper from saved tensors in `figures_for_main_doc/`.
- **`runner_for_overleaf_figures.ipynb`** – builds tables and figures for the supplementary information from experiment outputs under `experiments_zeta=-0.14/`.

### Scripts: `network_generator/overleaf_figures/`

- `generate_networks_for_overleaf_figures.py` – batch-generate networks and tabulate properties.
- `visualize_for_overleaf_figures.py` – plotting helpers for main-document and S.I. layouts.
- `diagnostics_latex_table.py`, `best_theta_parameters_latex_table.py`, `robustnesscheck_pos_neg_roundneg_latex_table.py` – convert diagnostic / θ CSV results into LaTeX tables.

## Paper artifacts and supporting CSVs

### `figures_for_main_doc/`

Saved network tensors (`.pth`) and example plots (`.png`) used to produce the tables and figures in the **main research paper** (driven by `runner_for_overleaf_maindoc.ipynb`).

### `experiments_zeta=-0.14/`

Compiled diagnostics CSVs and summary plots from replication experiments (Hungary at several network sizes; different country runs, and θ-variant robustness checks). These are the inputs needed to produce the tables and figures in the **supplementary information** (driven by `runner_for_overleaf_figures.ipynb`).

### Root-level CSV files

The project-root `.csv` files are used by notebooks under `additional_notebooks/`:

| File | Typical use |
|------|-------------|
| `correlation_results.csv` | ζ reordering / correlation sweeps (`runner_tuning_zeta_for_reordering_adjmat.ipynb`) |
| `dataframe_of_best_thetas_smaller_alpha_nu_to_compensate_and_zeta=-0.14_200expts.csv` | Best θ aggregates from batch Optuna |
| `dataframe_of_best_thetas_[-0.5,0.5]_[-1,-0.5]_[-1.5,-1]_[-1.5,-1]_smaller_alpha_nu_to_compensate_and_zeta=-0.14_200expts.csv` | Best θ aggregates under restricted θ bounds (ζ = −0.14) |

## Project structure

```
synthetic_supply_networks/
├── conf/
│   ├── network/default.yaml
│   ├── target/default.yaml
│   └── training/default.yaml
├── data/                          # Country IOT .pkl files (e.g. HUN.pkl)
├── network_generator/
│   ├── __init__.py
│   ├── config.py                  # NetworkConfig, TargetConfig, TrainingConfig
│   ├── data_iot.py                # Load IOT, assign firms to industries
│   ├── initialization.py          # Burr degrees, reordering, configuration model, weights
│   ├── metrics.py                 # Network statistics and loss-related quantities
│   ├── diagnostics.py             # compute_network_stats → DataFrame
│   ├── visualization.py           # Network / IOT visualization
│   ├── utils.py                   # firm_model_to_iot, scale_by_iot
│   ├── optuna_search/
│   │   ├── search_algo_fourparams.py   # optuna_search_single_experiment
│   │   └── run_multiple_times.py       # run_optuna_theta_multiple_experiments
│   └── overleaf_figures/
│       ├── generate_networks_for_overleaf_figures.py
│       ├── visualize_for_overleaf_figures.py
│       ├── diagnostics_latex_table.py
│       ├── best_theta_parameters_latex_table.py
│       └── robustnesscheck_pos_neg_roundneg_latex_table.py
├── additional_notebooks/          # Optuna batch, ζ tuning, Burr grids, Overleaf runners
├── figures_for_main_doc/          # Tensors/plots for main-paper figures
├── experiments_zeta=-0.14/        # Experiment CSVs/plots for SI figures
├── *.csv                          # Supporting CSVs for additional_notebooks
├── runner.ipynb                   # Full interactive workflow
└── README.md
```

## Notes

- **Sparse tensors**: The pipeline uses PyTorch sparse COO matrices; very large runs need sufficient RAM and appropriate device usage.
- **Reproducibility**: Firm-to-industry assignment and stochastic degree draws vary unless you fix random seeds everywhere they are used.
- **TrainingConfig naming**: Despite the name, `TrainingConfig` configures **Optuna** (trial counts and θ bounds), not epoch-based neural network training.
- **θ sources**: After Optuna, use the returned best θ in `populate_weights_fourparams` if you want the optimized network; the default `runner.ipynb` cell may still use YAML defaults unless you decide to use the results from `best_optuna_stats`.
