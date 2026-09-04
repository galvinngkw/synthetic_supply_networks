# Synthetic Supply Networks

A PyTorch-based generator for firm-level supply networks calibrated to empirical input–output tables and distributional targets (degrees, strengths, correlations, Hill exponents, and related statistics).

## Overview

This repository builds synthetic supply networks by:

- **Heavy-tailed degrees (Burr XII)**: In- and out-degree sequences are drawn from truncated Burr distributions; parameters are chosen so that mean, log-variance, and Hill exponent match configuration targets.
- **Degree reordering**: `triangular_wedge_matching` reorders degrees; `noise_amount_for_reordering_burrkiko` in `conf/network/default.yaml` controls the wedge noise ($\zeta$; default −0.14).
- **Sparse topology**: A directed configuration model (`igraph`) realizes the degree sequences by sampling an adjacency matrix. 
- **Lognormal fitnesses**: Initial in- and out-fitnesses sequences are correlated lognormals. 
- **Four-parameter edge weights**: Weights use `populate_weights_fourparams` with $\theta_{k,\mathrm{in}}, \theta_{k,\mathrm{out}},\theta_{f,\mathrm{in}}, \theta_{f,\mathrm{out}}$ (defaults in YAML; optimizable via Optuna).
- **IOT consistency**: `scale_by_iot` rescales the firm–firm network so its industry-level aggregate matches the country’s input–output table (`data_iot.load_input_output_table`).
- **Optuna tuning**: `TrainingConfig` drives Optuna search over the four $\theta$ bounds and trial count (`conf/training/default.yaml`). The single-run API is `optuna_search_single_experiment` in `network_generator/optuna_search/search_algo_fourparams.py`; batch replication uses `run_optuna_theta_multiple_experiments` in `run_multiple_times.py`.

**Metrics and diagnostics**: `network_generator/metrics.py` implements degrees, strengths, Hill exponents, OLS/TLS regressions, PageRank, correlations, and related quantities. `diagnostics.compute_network_stats` returns a wide property table for a weighted network.

## Installation

Dependencies are declared in `pyproject.toml` (Python ≥ 3.12). The recommended installer is [`uv`](https://docs.astral.sh/uv/); see the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) if you do not have it yet.

```bash
git clone <repository-url>
cd synthetic_supply_networks
uv sync --group dev --group paper
```

- `uv sync` installs the core scientific stack and the editable `network_generator` package.
- `--group dev` adds Jupyter tooling (`ipykernel`, `jupyterlab`, `notebook`) for `runner.ipynb` and the notebooks under `paper/notebooks/`.
- `--group paper` adds `seaborn` and `SciencePlots` for Overleaf figure helpers under `paper/scripts/`.

Without `uv`, you can install with pip from the same `pyproject.toml`:

```bash
pip install -e .
pip install ipykernel jupyterlab notebook seaborn SciencePlots
```

Key packages and their roles:

- **torch**: sparse tensors throughout the pipeline
- **scipy**: Burr parameter solving (`optimize.root_scalar`, `stats.burr12`)
- **igraph**: configuration model and compute network properties
- **optuna**: $\theta$ hyperparameter search (TPE sampler)
- **tqdm**: progress bars in batch Optuna runs and Overleaf batch scripts
- **pydantic** / **PyYAML**: typed configs loaded from `conf/`
- **seaborn** / **SciencePlots**: publication plotting in the Overleaf helpers (`science` / `nature` styles)

### Data

This project uses input-output tables from the OECD Inter-Country
Input-Output (ICIO) database, 2021 edition.

The OECD tables used by the project are stored under [`data/`](data/) as
Python pickle (`.pkl`) files named `{COUNTRY_CODE}.pkl`, such as
`HUN.pkl`. Each file maps a year to an industry-by-industry input-output
matrix measured in millions of USD. The loader rescales the values using
`monetary_unit` in `NetworkConfig`; see
`data_iot.load_input_output_table`.

The default configuration in `conf/network/default.yaml` uses Hungary
(`HUN`) for 2015. To use another country or year, ensure that the
corresponding `data/{COUNTRY_CODE}.pkl` file contains the requested table.

The `.pkl` files contain OECD data or reformatted OECD data. Converting the
tables to Python pickle format does not change the terms applicable to the
underlying data. These files are not covered by the repository's MIT
License.

See [`data/README.md`](data/README.md) for file-level provenance and
[`DATA_NOTICE.md`](DATA_NOTICE.md) for attribution and applicable data
terms.

> **Security:** Python pickle files can execute code when loaded. Only load
> `.pkl` files obtained from a trusted source.

## Configuration

Default parameters live in YAML files under `conf/` and are loaded automatically by `NetworkConfig`, `TargetConfig`, and `TrainingConfig` in `network_generator/config.py`. **Edit these files** to change shared defaults (e.g. country, number of firms, Optuna trial count):

| File | Role |
|------|------|
| `conf/network/default.yaml` | Firms count, country/year, monetary unit, Burr targets and cutoffs, reordering noise $\zeta$, lognormal strength parameters, default $\theta$ values, Hill percentile thresholds |
| `conf/target/default.yaml` | Targets for optimization loss (Hill exponents, moments, correlations, regression slopes, TLS terms) |
| `conf/training/default.yaml` | Optuna: `num_experiments`, `n_trials`, and box constraints for each $\theta$ |

You can also **override YAML defaults in the notebook** by passing arguments when constructing the config objects (arguments take precedence over the YAML files):

```python
network_config = NetworkConfig(num_firms=100_000, country_code="BEL")
target_config = TargetConfig()
training_config = TrainingConfig(n_trials=300)
```

If you change `country_code`, ensure the matching IOT pickle exists under `data/` (e.g. `data/BEL.pkl`).

## Running the project

### Main notebook: `runner.ipynb`

**Quick start:** Open `runner.ipynb` and run every cell from top to bottom. The notebook is self-contained—follow the markdown section headers and inline comments. With default settings you get a full synthetic network (degree sampling → topology → Optuna tuning → IOT-scaled weights → diagnostics and plots). You do not need to edit any code to try it once.

**Changing parameters:** Use the config cell near the top of the notebook (see [Configuration](#configuration) above). Either edit the YAML files under `conf/`, or pass overrides to `NetworkConfig()`, `TargetConfig()`, and `TrainingConfig()` as shown there.

**Pipeline overview** (what the notebook runs, in order):

1. Load configs and target tensors.
2. Sample Burr in-/out-degree sequences; reorder with `triangular_wedge_matching`.
3. Build topology with the directed configuration model; generate initial fitnesses.
4. Load the input–output table and assign firms to industries (proportional to industry sales; rerunning changes the firm–industry map).
5. Run Optuna to tune the four $\theta$ parameters; inspect trial results and plots.
6. Build the final weighted network with `populate_weights_fourparams` and `scale_by_iot` (by default using $\theta$ from `network_config`; you can substitute values from `best_optuna_stats` instead).
7. Run diagnostics and visualizations (`compute_network_stats`, `visualize_network`, heatmaps, IOT comparison, CCDFs).

Degree resampling loops and Optuna with many trials can be slow for large `num_firms`.

### Batch Optuna: `paper/notebooks/runner_for_optuna_multiple_times.ipynb`

Uses `TrainingConfig` and `run_optuna_theta_multiple_experiments` to repeat the random network draw + Optuna many times and aggregate best $\theta$ statistics (see `network_generator/optuna_search/run_multiple_times.py`).

### Other notebooks (under `paper/notebooks/`)

- **`runner_for_systematic_tuning_of_params.ipynb`** – grids over Burr-related controls (e.g. $\alpha$, $\nu$) for degree statistics, and ablation studies.
- **`runner_tuning_zeta_for_reordering_adjmat.ipynb`** – explores the reordering noise parameter $\zeta$.
- **`runner_for_overleaf_maindoc.ipynb`** – builds figures for the main research paper from saved tensors in `paper/artifacts/main_paper/`.
- **`runner_for_overleaf_figures.ipynb`** – builds S.I. tables and figures from experiment outputs under `paper/artifacts/supplementary/experiments_zeta=-0.14/`. Use `generate_multiple_networks_and_save_them` to generate networks and `visualize_network_for_overleaf_si` to plot them. To reproduce Section D of the S.I. from the saved diagnostics, run from the section “Run the remaining cells to reproduce tables and figures in the S.I. Section D Robustness checks” onward (imports and configs above that section are still required).

### Scripts: `paper/scripts/`

- `generate_networks_for_overleaf_figures.py` – batch-generate networks and tabulate properties.
- `visualize_for_overleaf_figures.py` – plotting helpers for main-document and S.I. layouts.
- `diagnostics_latex_table.py`, `best_theta_parameters_latex_table.py`, `robustnesscheck_pos_neg_roundneg_latex_table.py` – convert diagnostic / $\theta$ CSVs results into LaTeX tables.

## Paper artifacts

All publication-related notebooks, scripts, and saved outputs live under `paper/`:

### `paper/artifacts/main_paper/`

Saved network tensors (`.pth`) and example plots (`.png`) used to produce the tables and figures in the **main research paper** (driven by `runner_for_overleaf_maindoc.ipynb`).

### `paper/artifacts/supplementary/experiments_zeta=-0.14/`

Compiled diagnostics CSVs and summary plots from replication experiments (Hungary at several network sizes; different country runs, and $\theta$-variant robustness checks). These are the inputs needed to produce the tables and figures in the **supplementary information** (driven by `runner_for_overleaf_figures.ipynb`).

### `paper/artifacts/tuning_csvs/`

Intermediate CSV outputs from parameter-tuning notebooks:

| File | Typical use |
|------|-------------|
| `zeta_reordering_correlation_sweep.csv` | $\zeta$ reordering / correlation sweeps (`runner_tuning_zeta_for_reordering_adjmat.ipynb`) |
| `best_thetas_200expts_full_bounds_zeta-0.14.csv` | Best $\theta$ aggregates from batch Optuna (full search bounds; reported in S.I.) |
| `best_thetas_200expts_restricted_bounds_zeta-0.14.csv` | Best $\theta$ aggregates under restricted $\theta$ bounds ($\zeta$ = −0.14) |

## Project structure

```
synthetic_supply_networks/
├── conf/
│   ├── network/default.yaml
│   ├── target/default.yaml
│   └── training/default.yaml
├── data/                          # Country IOT .pkl files (e.g. HUN.pkl)
│   └── README.md                   # Data format, provenance, and usage terms
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
├── paper/
│   ├── notebooks/                      # Optuna batch, ζ tuning, Burr grids, Overleaf runners
│   ├── scripts/                        # LaTeX tables and publication plotting helpers
│   └── artifacts/
│       ├── main_paper/              # Tensors/plots for main-paper figures
│       ├── supplementary/              # Experiment CSVs/plots for SI figures
│       └── tuning_csvs/                # Intermediate tuning outputs
├── pyproject.toml                 # Project metadata and dependencies
├── DATA_NOTICE.md                 # OECD data licensing and attribution
├── LICENSE                         # MIT License for the source code
├── runner.ipynb                   # Full interactive workflow
└── README.md
```

## Notes

- **Sparse tensors**: The pipeline uses PyTorch sparse COO matrices; very large runs need sufficient RAM and appropriate device usage.
- **Reproducibility**: Firm-to-industry assignment and stochastic degree draws vary unless you fix random seeds everywhere they are used.
- **TrainingConfig naming**: Despite the name, `TrainingConfig` configures **Optuna** (trial counts and $\theta$ bounds), not epoch-based neural network training.
- **$\theta$ sources**: After Optuna, use the returned best $\theta$ in `populate_weights_fourparams` if you want the optimized network; the default `runner.ipynb` cell may still use YAML defaults unless you decide to use the results from `best_optuna_stats`.

## License

Except for third-party materials and data or data-derived artifacts identified
in this repository, the source code is licensed under the
[MIT License](LICENSE).

The MIT License does not apply to the OECD input-output table data stored
under [`data/`](data/), including the `.pkl` files derived from those tables.
The OECD data remains subject to the applicable OECD terms, attribution
requirements, dataset-specific restrictions, and any relevant third-party
rights.

See [`DATA_NOTICE.md`](DATA_NOTICE.md) for data licensing and attribution
information and [`data/README.md`](data/README.md) for file-level provenance.
