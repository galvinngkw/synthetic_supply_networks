"""Function to visualize network distributions"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import BoundaryNorm, LogNorm
from network_generator.metrics import (
    compute_degrees_from_model,
    compute_pagerank,
    compute_strengths_from_model,
)
from network_generator.utils import firm_model_to_iot


def series_to_ccdf_dict(x: pd.Series):
    pdf = x.value_counts().sort_index() / len(x)
    x = pdf.index.to_numpy()
    ccdf_values = pdf[::-1].cumsum()[::-1].to_numpy()
    return x, ccdf_values


def visualize_network(
    model: torch.Tensor,
    firm_to_industry_tensor: torch.Tensor,
    data_iot: torch.Tensor,
    monetary_unit: float,
    country_code: str,
    country_year: int,
):
    """Visualize the network.

    Args:
        model: The network model.
        firm_to_industry_tensor: The firm to industry tensor.
        data_iot: The data IOT tensor.
        monetary_unit: The monetary unit.
        country_code: The country code.
        country_year: The country year.
    """

    fig, axes = plt.subplots(5, 2, figsize=(14, 24))

    in_deg, out_deg = compute_degrees_from_model(model)
    _, edge_non_neg_weights, in_str, out_str = compute_strengths_from_model(model)
    edge_pos_weights = edge_non_neg_weights[edge_non_neg_weights > 0]
    pagerank = compute_pagerank(model)

    num_industries = data_iot.size(dim=0)
    data_iot_np = data_iot.detach().numpy()
    output_agg_iot = firm_model_to_iot(model, firm_to_industry_tensor, num_industries)
    output_agg_iot_np = output_agg_iot.detach().numpy()

    in_deg_pd = pd.Series(in_deg.detach().numpy()).round().astype(int)
    out_deg_pd = pd.Series(out_deg.detach().numpy()).round().astype(int)
    in_str_pd = pd.Series(in_str.detach().numpy() * monetary_unit)
    out_str_pd = pd.Series(out_str.detach().numpy() * monetary_unit)
    pagerank_pd = pd.Series(pagerank.detach().numpy())
    edge_pos_weights_pd = pd.Series(edge_pos_weights.detach().numpy() * monetary_unit)

    # Compute quantiles across both datasets for consistent colormap
    all_data = np.concatenate(
        [output_agg_iot_np.flatten(), data_iot_np.flatten()]
    )  # Combine both datasets
    quantiles = np.quantile(
        all_data, [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    )

    # Create a colormap and a BoundaryNorm based on the quantiles
    cmap = plt.cm.viridis
    norm = BoundaryNorm(quantiles, ncolors=cmap.N, clip=True)

    # Panel 1 (Row 0, Col 0): Heatmap of output aggregated IOT
    img1 = axes[0, 0].imshow(output_agg_iot_np, cmap=cmap, norm=norm, aspect="auto")
    axes[0, 0].set_title(
        f"Aggregated synthetic network, quantile-binned ({monetary_unit/1e6:.0f} million USD)"
    )

    # Panel 2 (Row 0, Col 1): Heatmap of true IOT
    img2 = axes[0, 1].imshow(data_iot_np, cmap=cmap, norm=norm, aspect="auto")
    axes[0, 1].set_title(
        f"Empirical IOT ({str(country_code)} {str(country_year)}), quantile-binned ({monetary_unit/1e6:.0f} million USD)"
    )

    # Shared colorbar
    fig.colorbar(
        img2,
        ax=axes[0, 1],
        orientation="vertical",
        pad=0.02,
        fraction=0.1,
        format="%.3g",
    )

    # Panel 3 (Row 1, Col 0): CCDF of pagerank
    x, y = series_to_ccdf_dict(pagerank_pd)
    axes[1, 0].loglog(x, y)
    axes[1, 0].set_title("CCDF of pagerank")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_ylabel(r"$P(X\geq x)$")

    # Panel 4 (Row 1, Col 1): CCDF of edge weights
    x, y = series_to_ccdf_dict(edge_pos_weights_pd)
    axes[1, 1].loglog(x[x > 0], y[x > 0])
    axes[1, 1].set_title("CCDF of edge weights (1 USD)")
    axes[1, 1].set_ylabel(r"$P(X\geq x)$")

    # Panel 5 (Row 2, Col 0): CCDF of degrees
    x_in, y_in = series_to_ccdf_dict(in_deg_pd)
    x_out, y_out = series_to_ccdf_dict(out_deg_pd)
    axes[2, 0].loglog(x_in[x_in > 0], y_in[x_in > 0], label="in-deg")
    axes[2, 0].loglog(x_out[x_out > 0], y_out[x_out > 0], label="out-deg")
    axes[2, 0].set_title("CCDF of degrees")
    axes[2, 0].set_yscale("log")
    axes[2, 0].set_xscale("log")
    axes[2, 0].set_ylabel(r"$P(X\geq x)$")
    axes[2, 0].legend()

    # Panel 6 (Row 2, Col 1): CCDF of strengths
    x_in_str, y_in_str = series_to_ccdf_dict(in_str_pd)
    x_out_str, y_out_str = series_to_ccdf_dict(out_str_pd)
    axes[2, 1].loglog(x_in_str[x_in_str > 0], y_in_str[x_in_str > 0], label="in-str")
    axes[2, 1].loglog(
        x_out_str[x_out_str > 0], y_out_str[x_out_str > 0], label="out-str"
    )
    axes[2, 1].set_title("CCDF of strength (1 USD)")
    axes[2, 1].set_yscale("log")
    axes[2, 1].set_xscale("log")
    axes[2, 1].set_ylabel(r"$P(X\geq x)$")
    axes[2, 1].legend()

    # Panel 7 (Row 3, Col 0): out-degree vs out-strength
    h = axes[3, 0].hist2d(
        x=out_deg_pd,
        y=out_str_pd,
        bins=[
            np.logspace(np.log10(0.7), np.log10(1e5), 100),
            np.logspace(np.log10(1e-4), np.log10(1e11), 100),
        ],
        norm=LogNorm(),
    )
    axes[3, 0].set_xscale("log")
    axes[3, 0].set_yscale("log")
    axes[3, 0].set_xlabel("out-degree")
    axes[3, 0].set_ylabel("out-strength (1 USD)")
    fig.colorbar(h[3], ax=axes[3, 0], label="Counts")

    # Panel 8 (Row 3, Col 1): in-degree vs in-strength
    h = axes[3, 1].hist2d(
        x=in_deg_pd,
        y=in_str_pd,
        bins=[
            np.logspace(np.log10(0.7), np.log10(1e4), 100),
            np.logspace(np.log10(1e-6), np.log10(1e11), 100),
        ],
        norm=LogNorm(),
    )
    axes[3, 1].set_xscale("log")
    axes[3, 1].set_yscale("log")
    axes[3, 1].set_xlabel("in-degree")
    axes[3, 1].set_ylabel("in-strength (1 USD)")
    fig.colorbar(h[3], ax=axes[3, 1], label="Counts")

    # Panel 9 (Row 4, Col 0): out-degree vs in-degree
    h = axes[4, 0].hist2d(
        x=out_deg_pd,
        y=in_deg_pd,
        bins=[
            np.logspace(np.log10(0.7), np.log10(1e5), 100),
            np.logspace(np.log10(0.7), np.log10(1e4), 100),
        ],
        norm=LogNorm(),
    )
    axes[4, 0].set_xscale("log")
    axes[4, 0].set_yscale("log")
    axes[4, 0].set_xlabel("out-degree")
    axes[4, 0].set_ylabel("in-degree")
    fig.colorbar(h[3], ax=axes[4, 0], label="Counts")

    # Panel 10 (Row 4, Col 1): out-strength vs in-strength
    h = axes[4, 1].hist2d(
        x=out_str_pd,
        y=in_str_pd,
        bins=[
            np.logspace(np.log10(1e-4), np.log10(1e11), 100),
            np.logspace(np.log10(1e-6), np.log10(1e11), 100),
        ],
        norm=LogNorm(),
    )
    axes[4, 1].set_xscale("log")
    axes[4, 1].set_yscale("log")
    axes[4, 1].set_xlabel("out-strength (1 USD)")
    axes[4, 1].set_ylabel("in-strength (1 USD)")
    fig.colorbar(h[3], ax=axes[4, 1], label="Counts")

    plt.tight_layout()
    # plt.grid(None) toggles grid on the *current* axes only (Matplotlib API); use explicit False.
    for ax in fig.axes:
        ax.grid(False)
    plt.show()
