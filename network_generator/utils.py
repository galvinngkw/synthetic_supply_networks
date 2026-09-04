"""Utility functions for network generation.

This module provides:
- Aggregate transforms (firm-level matrix to industry-level matrix, IOT scaling)
"""

import torch


# -----------------------------------------------------------------------------
# Aggregate transforms (firm-level matrix to industry-level matrix, IOT scaling)
# -----------------------------------------------------------------------------
def firm_model_to_iot(
    model: torch.Tensor,
    firm_to_industry: torch.Tensor,
    num_industries: int,
) -> torch.Tensor:
    """Aggregate firm-level sparse matrix to industry-level IOT.

    Args:
        model: Sparse COO firm-firm weight matrix.
        firm_to_industry: Tensor mapping firm index to industry index.
        num_industries: Number of industries.

    Returns:
        Dense industry x industry IOT matrix.
    """
    model = model.coalesce()
    row_idx, col_idx = model.indices()
    edge_weights = model.values()

    row_ind = firm_to_industry[row_idx].long()
    col_ind = firm_to_industry[col_idx].long()

    flat_idx = row_ind * num_industries + col_ind
    agg_flat = torch.zeros(num_industries * num_industries, dtype=torch.float32)
    agg_flat = agg_flat.scatter_add(0, flat_idx, edge_weights)

    return agg_flat.view(num_industries, num_industries)


def scale_by_iot(
    model: torch.Tensor,
    target_iot: torch.Tensor,
    firm_to_industry: torch.Tensor,
) -> torch.Tensor:
    """Scale firm-firm weights to match industry-level IOT targets.

    Args:
        model: Sparse COO firm-firm weight matrix.
        target_iot: Target industry x industry IOT matrix.
        firm_to_industry: Tensor mapping firm index to industry index.

    Returns:
        Scaled sparse COO tensor.
    """
    device = model.device
    target_iot = target_iot.to(device)
    firm_to_industry = firm_to_industry.to(device)

    model = model.coalesce()
    row_idx, col_idx = model.indices()
    edge_weights = model.values()

    num_firms = model.size(0)
    num_ind = target_iot.size(0)

    row_ind = firm_to_industry[row_idx].long()
    col_ind = firm_to_industry[col_idx].long()
    flat_idx = row_ind * num_ind + col_ind

    transformed_flat = torch.zeros(
        num_ind * num_ind, device=device, dtype=edge_weights.dtype
    )
    transformed_flat.scatter_add_(0, flat_idx, edge_weights)
    transformed_model = transformed_flat.view(num_ind, num_ind)

    nonzero_mask = transformed_model != 0
    scaling_factors = torch.zeros_like(transformed_model)
    scaling_factors[nonzero_mask] = (
        target_iot[nonzero_mask] / transformed_model[nonzero_mask]
    )

    edge_scale = scaling_factors[row_ind, col_ind]
    new_edge_weights = edge_weights * edge_scale

    mask = new_edge_weights != 0
    if mask.any():
        return torch.sparse_coo_tensor(
            torch.stack((row_idx[mask], col_idx[mask])),
            new_edge_weights[mask],
            (num_firms, num_firms),
            device=device,
        ).coalesce()
    else:
        return torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long, device=device),
            torch.empty(0, dtype=edge_weights.dtype, device=device),
            (num_firms, num_firms),
        )
