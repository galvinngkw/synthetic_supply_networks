"""
Load Input-Output Table data from pickle file.
Assign firms to industries.
"""

import pickle
from pathlib import Path
import random
import numpy as np
import torch


def load_input_output_table(
    data_path: str | Path,
    country_code: str,
    year: int,
    monetary_unit: float,
) -> tuple[torch.Tensor, np.ndarray]:
    """Load Input-Output Table data from pickle file.

    Args:
        data_path: Path to the data directory or pickle file.
        country_code: Country code (e.g., "BEL").
        year: Year of the IOT data.
        monetary_unit: Unit for monetary values (default 100M USD).

    Returns:
        Tuple of (iot_tensor, iot_numpy) with rescaled values.
    """
    data_path = Path(data_path)

    if data_path.is_file():
        pkl_path = data_path
    else:
        pkl_path = data_path / f"{country_code}.pkl"

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # OECD-ICIO data is in 1 million USD, convert to monetary_unit
    iot_numpy = data[year] * 1e6 / monetary_unit
    iot_tensor = torch.tensor(iot_numpy, dtype=torch.float32)
    num_industries = iot_tensor.size(dim=0)

    return iot_tensor, iot_numpy, num_industries


# -----------------------------------------------------------------------------
# Assign firms to industries
# -----------------------------------------------------------------------------


def assign_firms_to_industries(
    num_firms: int,
    iot_data: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Assign firms to industries proportional to industry sales.

    Args:
        num_firms: Total number of firms.
        iot_data: Industry x industry IOT matrix.

    Returns:
        Tensor mapping firm index to industry index.
    """
    if isinstance(iot_data, torch.Tensor):
        iot_data = iot_data.numpy()

    # Compute industry shares based on total sales
    industry_sales = np.sum(iot_data, axis=1)
    total_sales = np.sum(industry_sales)
    shares = industry_sales / total_sales

    # Assign firms proportionally
    num_firms_per_industry = np.round(num_firms * shares).astype(int)
    num_industries = len(num_firms_per_industry)

    # Adjust to match exact num_firms
    diff = num_firms - num_firms_per_industry.sum()
    abs_diff = abs(diff)

    while abs_diff > 0:
        idx = random.randint(0, num_industries - 1)
        if num_firms_per_industry[idx] >= 1:
            num_firms_per_industry[idx] += np.sign(diff)
            abs_diff -= 1

    assert num_firms == num_firms_per_industry.sum(), "Firm count mismatch"

    # Create mapping tensor
    firm_to_industry = torch.from_numpy(
        np.repeat(
            np.arange(num_industries),
            num_firms_per_industry,
        )
    )

    return firm_to_industry
