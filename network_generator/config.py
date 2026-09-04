"""Load configuration from YAML files."""

from pathlib import Path
from typing import Annotated, Optional

import yaml
from pydantic import BaseModel, Field

DATA_PATH = Path(__file__).parents[1] / "data"
CONF_PATH = Path(__file__).parents[1] / "conf"

# Type aliases for common constraints
PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0)]
UnitInterval = Annotated[float, Field(ge=0, le=1)]
Correlation = Annotated[float, Field(ge=-1, le=1)]


class NetworkConfig(BaseModel, frozen=True):
    """Configuration for network structure and initialization."""

    num_firms: PositiveInt
    monetary_unit: PositiveFloat
    country_code: str
    country_year: PositiveInt

    # Burr degree generation
    burr_alpha_outdeg: PositiveFloat
    burr_var_log_outdeg: PositiveFloat
    burr_alpha_indeg: PositiveFloat
    burr_var_log_indeg: PositiveFloat
    burr_max_indeg: UnitInterval
    burr_max_outdeg: UnitInterval

    # Burr degree reordering
    noise_amount_for_reordering_burrkiko: float

    # Parameters for lognormal strength fitnesses generation
    mean_for_gen_lognorm_fi: float
    mean_for_gen_lognorm_fo: float
    corr_lognorm_gen_fi_fo: Correlation

    # Weight population parameters
    theta_k_in_value: float
    theta_k_out_value: float
    theta_f_in_value: float
    theta_f_out_value: float

    # Percentile thresholds for computing Hill exponents
    threshold_percentile_for_hill_deg: UnitInterval
    threshold_percentile_for_hill_strength: UnitInterval
    threshold_percentile_for_hill_weight: UnitInterval
    threshold_percentile_for_hill_influence: UnitInterval

    def __init__(self, **kwargs):
        """Initialize config, loading from YAML and merging with provided kwargs.

        Args:
            **kwargs: Configuration values to override YAML defaults.
                     Use _load_from_yaml=False to prevent auto-loading from YAML.
        """
        _load_from_yaml = kwargs.pop("_load_from_yaml", True)
        if _load_from_yaml:
            yaml_path = CONF_PATH / "network" / "default.yaml"
            if not yaml_path.exists():
                raise FileNotFoundError(f"YAML config file not found: {yaml_path}")
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f) or {}
            # Merge YAML data with kwargs (kwargs take precedence)
            data.update(kwargs)
            kwargs = data
        super().__init__(**kwargs)

    @classmethod
    def from_yaml(cls, yaml_path: Optional[Path] = None) -> "NetworkConfig":
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML file. If None, uses conf/network/default.yaml

        Returns:
            NetworkConfig instance loaded from YAML
        """
        if yaml_path is None:
            yaml_path = CONF_PATH / "network" / "default.yaml"

        if yaml_path.exists():
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f) or {}
            # Use __init__ to ensure proper initialization path
            return cls(_load_from_yaml=False, **data)
        else:
            raise FileNotFoundError(f"YAML config file not found: {yaml_path}")


class TargetConfig(BaseModel, frozen=True):
    """Target network properties for optimization."""

    # Hill exponents
    target_hill_in_deg: PositiveFloat
    target_hill_out_deg: PositiveFloat
    target_hill_weight_distribution: PositiveFloat
    target_hill_in_str: PositiveFloat
    target_hill_out_str: PositiveFloat

    # Average connection
    target_average_connection: PositiveFloat
    target_variance_si: PositiveFloat
    target_variance_so: PositiveFloat

    # Correlations
    target_so_ko_corr: Correlation
    target_si_ki_corr: Correlation
    target_si_so_corr: Correlation
    target_ki_ko_corr: Correlation

    # Regression coefficients
    target_so_ko_regcoeff: float
    target_ko_so_regcoeff: float
    target_si_ki_regcoeff: float
    target_ki_si_regcoeff: float

    # TLS targets
    target_tls_str: float
    target_tls_deg: float

    def __init__(self, **kwargs):
        """Initialize config, loading from YAML and merging with provided kwargs.

        Args:
            **kwargs: Configuration values to override YAML defaults.
                     Use _load_from_yaml=False to prevent auto-loading from YAML.
        """
        _load_from_yaml = kwargs.pop("_load_from_yaml", True)
        if _load_from_yaml:
            yaml_path = CONF_PATH / "target" / "default.yaml"
            if not yaml_path.exists():
                raise FileNotFoundError(f"YAML config file not found: {yaml_path}")
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f) or {}
            # Merge YAML data with kwargs (kwargs take precedence)
            data.update(kwargs)
            kwargs = data
        super().__init__(**kwargs)

    @classmethod
    def from_yaml(cls, yaml_path: Optional[Path] = None) -> "TargetConfig":
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML file. If None, uses conf/target/default.yaml

        Returns:
            TargetConfig instance loaded from YAML
        """
        if yaml_path is None:
            yaml_path = CONF_PATH / "target" / "default.yaml"

        if yaml_path.exists():
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f) or {}
            # Use __init__ to ensure proper initialization path
            return cls(_load_from_yaml=False, **data)
        else:
            raise FileNotFoundError(f"YAML config file not found: {yaml_path}")


class TrainingConfig(BaseModel, frozen=True):
    """Configuration for the Optuna search process."""

    num_experiments: PositiveInt
    n_trials: PositiveInt
    theta_k_in_upper_bound: float
    theta_k_in_lower_bound: float
    theta_k_out_upper_bound: float
    theta_k_out_lower_bound: float
    theta_f_in_upper_bound: float
    theta_f_in_lower_bound: float
    theta_f_out_upper_bound: float
    theta_f_out_lower_bound: float

    def __init__(self, **kwargs):
        """Initialize config, loading from YAML and merging with provided kwargs.

        Args:
            **kwargs: Configuration values to override YAML defaults.
                        Use _load_from_yaml=False to prevent auto-loading from YAML.
        """
        _load_from_yaml = kwargs.pop("_load_from_yaml", True)
        if _load_from_yaml:
            yaml_path = CONF_PATH / "training" / "default.yaml"
            if not yaml_path.exists():
                raise FileNotFoundError(f"YAML config file not found: {yaml_path}")
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f) or {}
            # Merge YAML data with kwargs (kwargs take precedence)
            data.update(kwargs)
            kwargs = data
        super().__init__(**kwargs)

    @classmethod
    def from_yaml(cls, yaml_path: Optional[Path] = None) -> "TrainingConfig":
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML file. If None, uses conf/training/default.yaml

        Returns:
            TrainingConfig instance loaded from YAML
        """
        if yaml_path is None:
            yaml_path = CONF_PATH / "training" / "default.yaml"

        if yaml_path.exists():
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f) or {}
            # Use __init__ to ensure proper initialization path
            return cls(_load_from_yaml=False, **data)
        else:
            raise FileNotFoundError(f"YAML config file not found: {yaml_path}")
