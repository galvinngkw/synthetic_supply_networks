"""
Load input-output tables (local, cached or from the mirror).
Assign firms to industries.
"""

import pickle
from pathlib import Path
import random
import numpy as np
import torch

from network_generator.oecd_icio.store import DEFAULT_RELEASE, TableStore


def load_input_output_table(
    data_path: str | Path | None = "data",
    country_code: str = "HUN",
    year: int = 2015,
    monetary_unit: float = 1e8,
    release: str = DEFAULT_RELEASE,
) -> tuple[torch.Tensor, np.ndarray, int]:
    """Load a country's industry x industry input-output table.

    The table is looked up in ``data_path`` first (a folder of ``{COUNTRY}.pkl`` files, or one such file), then in the local cache, then downloaded from the pinned mirror of ``release`` (see ``network_generator.oecd_icio.store``). Pass ``data_path=None`` to skip the local folder.

    Args:
        data_path: Folder of ``{COUNTRY}.pkl`` files, a single pickle, or None.
        country_code: ISO3 country code (e.g., "BEL").
        year: Year of the IOT data.
        monetary_unit: Unit for monetary values (default 100M USD).
        release: OECD ICIO release used when the table is not found locally.

    Returns:
        Tuple of (iot_tensor, iot_numpy, num_industries) with values in ``monetary_unit``.
    """
    if data_path is not None and Path(data_path).is_file():
        with open(data_path, "rb") as f:
            table = pickle.load(f)[year]
    else:
        table = TableStore(release).get(country_code, year, data_dir=data_path)

    # OECD-ICIO data is in 1 million USD, convert to monetary_unit
    iot_numpy = np.asarray(table, dtype=np.float32) * 1e6 / monetary_unit
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
