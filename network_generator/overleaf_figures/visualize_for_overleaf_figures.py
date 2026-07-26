"""
Visualize the distributional properties of multiple networks, save them as .png files.
Also plot selected network properties as boxplots.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LogNorm
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import gridspec
from matplotlib.lines import Line2D
from tqdm import tqdm

from network_generator.metrics import (
    compute_degrees_from_model,
    compute_pagerank,
    compute_strengths_from_model,
)

from network_generator.utils import firm_model_to_iot

from network_generator.visualization import series_to_ccdf_dict


def visualize_deg_ccdf(
    model: torch.Tensor,
    save_path: str = None,
):
    in_deg, out_deg = compute_degrees_from_model(model)
    in_deg_pd = pd.Series(in_deg.detach().numpy()).round().astype(int)
    out_deg_pd = pd.Series(out_deg.detach().numpy()).round().astype(int)
    x_in, y_in = series_to_ccdf_dict(in_deg_pd)
    x_out, y_out = series_to_ccdf_dict(out_deg_pd)
    plt.loglog(x_in[x_in > 0], y_in[x_in > 0], label="in-deg")
    plt.loglog(x_out[x_out > 0], y_out[x_out > 0], label="out-deg")
    plt.title("CCDF of degrees", fontsize=15)
    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel(r"$P(X\geq x)$", fontsize=15)
    plt.legend(fontsize=15)
    if save_path is not None:
        plt.savefig(save_path, dpi="figure", bbox_inches="tight")
    plt.show()


def visualize_iot(
    model: torch.Tensor,
    firm_to_industry_tensor: torch.Tensor,
    monetary_unit: float,
    data_iot_np: np.ndarray,
    num_industries: int,
    country_code: str,
    country_year: int,
    save_path: str = None,
):
    output_agg_iot = firm_model_to_iot(model, firm_to_industry_tensor, num_industries)
    output_agg_iot_np = output_agg_iot.detach().numpy()
    # Compute quantiles across both datasets for consistent colormap
    all_data = np.concatenate(
        [output_agg_iot_np.flatten(), data_iot_np.flatten()]
    )  # Combine both datasets
    quantiles = np.quantile(
        all_data, [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))

    # Create a colormap and a BoundaryNorm based on the quantiles
    cmap = plt.cm.viridis
    norm = BoundaryNorm(quantiles, ncolors=cmap.N, clip=True)

    # Panel 1 (Row 0, Col 0): Heatmap of output aggregated IOT
    img1 = axes[0].imshow(output_agg_iot_np, cmap=cmap, norm=norm, aspect="auto")
    axes[0].set_title(
        f"Aggregated synthetic network",
        fontsize=12,
    )

    # Panel 2 (Row 0, Col 1): Heatmap of true IOT
    img2 = axes[1].imshow(data_iot_np, cmap=cmap, norm=norm, aspect="auto")
    axes[1].set_title(
        f"Empirical IOT ({str(country_code)} {str(country_year)})",
        fontsize=12,
    )

    # Shared colorbar
    cbar = fig.colorbar(
        img1,
        ax=[axes[0], axes[1]],
        orientation="horizontal",
        pad=0.13,
        fraction=0.05,
        format="%.3g",
    )
    cbar.ax.set_title(
        f"Monetary units: {monetary_unit/1e6:.0f} million USD",
        fontsize=11,
        pad=8,
    )

    if save_path is not None:
        plt.savefig(save_path, dpi="figure", bbox_inches="tight")

    plt.show()


def visualize_joint_distributions(
    model: torch.Tensor,
    monetary_unit: float,
    save_path: str = None,
):
    font_rc = {
        "font.size": 26,
        "axes.labelsize": 32,
        "xtick.labelsize": 26,
        "ytick.labelsize": 26,
    }

    with plt.rc_context(font_rc):
        fig, axes = plt.subplots(1, 4, figsize=(35, 7))

        in_deg, out_deg = compute_degrees_from_model(model)
        _, _, in_str, out_str = compute_strengths_from_model(model)
        in_deg_pd = pd.Series(in_deg.detach().numpy()).round().astype(int)
        out_deg_pd = pd.Series(out_deg.detach().numpy()).round().astype(int)
        in_str_pd = pd.Series(in_str.detach().numpy() * monetary_unit)
        out_str_pd = pd.Series(out_str.detach().numpy() * monetary_unit)

    # Panel 1: out-degree vs out-strength
    h = axes[0].hist2d(
        x=out_deg_pd,
        y=out_str_pd,
        bins=[
            np.logspace(np.log10(0.7), np.log10(1e5), 100),
            np.logspace(np.log10(1e-4), np.log10(1e11), 100),
        ],
        norm=LogNorm(),
    )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("out-degree")
    axes[0].set_ylabel("out-strength (1 USD)")
    cbar = fig.colorbar(h[3], ax=axes[0])
    cbar.ax.tick_params(labelsize=24)
    cbar.set_label("Counts", fontsize=28)

    # Panel 2: in-degree vs in-strength
    h = axes[1].hist2d(
        x=in_deg_pd,
        y=in_str_pd,
        bins=[
            np.logspace(np.log10(0.7), np.log10(1e4), 100),
            np.logspace(np.log10(1e-6), np.log10(1e11), 100),
        ],
        norm=LogNorm(),
    )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("in-degree")
    axes[1].set_ylabel("in-strength (1 USD)")
    cbar = fig.colorbar(h[3], ax=axes[1])
    cbar.ax.tick_params(labelsize=24)
    cbar.set_label("Counts", fontsize=28)

    # Panel 3: out-degree vs in-degree
    h = axes[2].hist2d(
        x=out_deg_pd,
        y=in_deg_pd,
        bins=[
            np.logspace(np.log10(0.7), np.log10(1e5), 100),
            np.logspace(np.log10(0.7), np.log10(1e4), 100),
        ],
        norm=LogNorm(),
    )
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("out-degree")
    axes[2].set_ylabel("in-degree")
    cbar = fig.colorbar(h[3], ax=axes[2])
    cbar.ax.tick_params(labelsize=24)
    cbar.set_label("Counts", fontsize=28)

    # Panel 4: in-strength vs out-strength
    h = axes[3].hist2d(
        x=out_str_pd,
        y=in_str_pd,
        bins=[
            np.logspace(np.log10(1e-4), np.log10(1e11), 100),
            np.logspace(np.log10(1e-6), np.log10(1e11), 100),
        ],
        norm=LogNorm(),
    )
    axes[3].set_xscale("log")
    axes[3].set_yscale("log")
    axes[3].set_xlabel("out-strength (1 USD)")
    axes[3].set_ylabel("in-strength (1 USD)")
    cbar = fig.colorbar(h[3], ax=axes[3])
    cbar.ax.tick_params(labelsize=24)
    cbar.set_label("Counts", fontsize=28)

    for ax in axes:
        ax.tick_params(axis="both", which="major", labelsize=26)
        ax.grid(False)

    if save_path is not None:
        plt.savefig(save_path, dpi="figure", bbox_inches="tight")

    plt.tight_layout()
    plt.show()


def visualize_network_for_overleaf_main(
    model: torch.Tensor,
    monetary_unit: float,
    save_path: str = None,
):
    """Visualize the network."""

    # ==========================================
    # Font size controls
    # ==========================================
    TITLE_FONTSIZE = 9
    LABEL_FONTSIZE = 8
    TICK_FONTSIZE = 7
    LEGEND_FONTSIZE = 7
    COLORBAR_LABEL_FONTSIZE = 7
    COLORBAR_TICK_FONTSIZE = 6
    PANEL_LABEL_FONTSIZE = 10

    # ==========================================
    # Colorbar layout controls
    # ==========================================
    COLORBAR_PAD = 0.08
    COLORBAR_FRACTION = 0.045

    label_style = {
        "fontsize": PANEL_LABEL_FONTSIZE,
        "fontweight": "bold",
        "ha": "right",
    }

    fig, axes = plt.subplots(2, 4, figsize=(9.5, 4.8), constrained_layout=True)

    in_deg, out_deg = compute_degrees_from_model(model)
    _, edge_non_neg_weights, in_str, out_str = compute_strengths_from_model(model)
    edge_pos_weights = edge_non_neg_weights[edge_non_neg_weights > 0]
    pagerank = compute_pagerank(model)

    in_deg_pd = pd.Series(in_deg.detach().numpy()).round().astype(int)
    out_deg_pd = pd.Series(out_deg.detach().numpy()).round().astype(int)
    in_str_pd = pd.Series(in_str.detach().numpy() * monetary_unit)
    out_str_pd = pd.Series(out_str.detach().numpy() * monetary_unit)
    pagerank_pd = pd.Series(pagerank.detach().numpy())
    edge_pos_weights_pd = pd.Series(edge_pos_weights.detach().numpy() * monetary_unit)

    # Row 0, Col 0: CCDF of degrees
    x_in, y_in = series_to_ccdf_dict(in_deg_pd)
    x_out, y_out = series_to_ccdf_dict(out_deg_pd)
    axes[0, 0].loglog(x_in[x_in > 0], y_in[x_in > 0], label="in-deg")
    axes[0, 0].loglog(x_out[x_out > 0], y_out[x_out > 0], label="out-deg")
    axes[0, 0].set_title("CCDF degrees", fontsize=TITLE_FONTSIZE)
    axes[0, 0].set_ylabel(r"$P(X\geq x)$", fontsize=LABEL_FONTSIZE)
    axes[0, 0].legend(fontsize=LEGEND_FONTSIZE)

    # Row 0, Col 1: CCDF of strengths
    x_in_str, y_in_str = series_to_ccdf_dict(in_str_pd)
    x_out_str, y_out_str = series_to_ccdf_dict(out_str_pd)
    axes[0, 1].loglog(x_in_str[x_in_str > 0], y_in_str[x_in_str > 0], label="in-str")
    axes[0, 1].loglog(
        x_out_str[x_out_str > 0], y_out_str[x_out_str > 0], label="out-str"
    )
    axes[0, 1].set_title("CCDF strengths (1 USD)", fontsize=TITLE_FONTSIZE)
    axes[0, 1].set_ylabel(r"$P(X\geq x)$", fontsize=LABEL_FONTSIZE)
    axes[0, 1].legend(fontsize=LEGEND_FONTSIZE)

    # Row 0, Col 2: CCDF of edge weights
    x, y = series_to_ccdf_dict(edge_pos_weights_pd)
    axes[0, 2].loglog(x[x > 0], y[x > 0])
    axes[0, 2].set_title("CCDF edge weights (1 USD)", fontsize=TITLE_FONTSIZE)
    axes[0, 2].set_ylabel(r"$P(X\geq x)$", fontsize=LABEL_FONTSIZE)

    # Row 0, Col 3: CCDF of pagerank
    x, y = series_to_ccdf_dict(pagerank_pd)
    axes[0, 3].loglog(x, y)
    axes[0, 3].set_title("CCDF pagerank", fontsize=TITLE_FONTSIZE)
    axes[0, 3].set_ylabel(r"$P(X\geq x)$", fontsize=LABEL_FONTSIZE)

    # Row 1, Col 0: out-degree vs out-strength
    h = axes[1, 0].hist2d(
        x=out_deg_pd,
        y=out_str_pd,
        bins=[
            np.logspace(np.log10(0.7), np.log10(1e5), 100),
            np.logspace(np.log10(1e-4), np.log10(1e11), 100),
        ],
        norm=LogNorm(),
    )
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("out-degree", fontsize=LABEL_FONTSIZE)
    axes[1, 0].set_ylabel("out-strength (1 USD)", fontsize=LABEL_FONTSIZE)

    cbar = fig.colorbar(
        h[3],
        ax=axes[1, 0],
        location="bottom",
        orientation="horizontal",
        pad=COLORBAR_PAD,
        fraction=COLORBAR_FRACTION,
    )
    cbar.set_label("Counts", fontsize=COLORBAR_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)

    # Row 1, Col 1: in-degree vs in-strength
    h = axes[1, 1].hist2d(
        x=in_deg_pd,
        y=in_str_pd,
        bins=[
            np.logspace(np.log10(0.7), np.log10(1e4), 100),
            np.logspace(np.log10(1e-6), np.log10(1e11), 100),
        ],
        norm=LogNorm(),
    )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("in-degree", fontsize=LABEL_FONTSIZE)
    axes[1, 1].set_ylabel("in-strength (1 USD)", fontsize=LABEL_FONTSIZE)

    cbar = fig.colorbar(
        h[3],
        ax=axes[1, 1],
        location="bottom",
        orientation="horizontal",
        pad=COLORBAR_PAD,
        fraction=COLORBAR_FRACTION,
    )
    cbar.set_label("Counts", fontsize=COLORBAR_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)

    # Row 1, Col 2: out-degree vs in-degree
    h = axes[1, 2].hist2d(
        x=out_deg_pd,
        y=in_deg_pd,
        bins=[
            np.logspace(np.log10(0.7), np.log10(1e5), 100),
            np.logspace(np.log10(0.7), np.log10(1e4), 100),
        ],
        norm=LogNorm(),
    )
    axes[1, 2].set_xscale("log")
    axes[1, 2].set_yscale("log")
    axes[1, 2].set_xlabel("out-degree", fontsize=LABEL_FONTSIZE)
    axes[1, 2].set_ylabel("in-degree", fontsize=LABEL_FONTSIZE)

    cbar = fig.colorbar(
        h[3],
        ax=axes[1, 2],
        location="bottom",
        orientation="horizontal",
        pad=COLORBAR_PAD,
        fraction=COLORBAR_FRACTION,
    )
    cbar.set_label("Counts", fontsize=COLORBAR_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)

    # Row 1, Col 3: out-strength vs in-strength
    h = axes[1, 3].hist2d(
        x=out_str_pd,
        y=in_str_pd,
        bins=[
            np.logspace(np.log10(1e-4), np.log10(1e11), 100),
            np.logspace(np.log10(1e-6), np.log10(1e11), 100),
        ],
        norm=LogNorm(),
    )
    axes[1, 3].set_xscale("log")
    axes[1, 3].set_yscale("log")
    axes[1, 3].set_xlabel("out-strength (1 USD)", fontsize=LABEL_FONTSIZE)
    axes[1, 3].set_ylabel("in-strength (1 USD)", fontsize=LABEL_FONTSIZE)

    cbar = fig.colorbar(
        h[3],
        ax=axes[1, 3],
        location="bottom",
        orientation="horizontal",
        pad=COLORBAR_PAD,
        fraction=COLORBAR_FRACTION,
    )
    cbar.set_label("Counts", fontsize=COLORBAR_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)

    # Panel labels
    panel_labels = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)"]

    for ax, label in zip(axes[0], panel_labels[:4]):
        ax.text(
            0.97,
            0.97,
            label,
            transform=ax.transAxes,
            va="top",
            **label_style,
        )

    # Second row: bottom-right
    for ax, label in zip(axes[1], panel_labels[4:]):
        ax.text(
            0.97,
            0.03,
            label,
            transform=ax.transAxes,
            va="bottom",
            **label_style,
        )

    # Apply tick font sizes and remove grids
    for ax in fig.axes:
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        ax.grid(False)

    if save_path is not None:
        plt.savefig(save_path, dpi="figure", bbox_inches="tight")

    plt.show()


def visualize_network_for_overleaf_si(
    num_experiments: int,
    num_firms: int,
    monetary_unit: float,
    country_code: str,
    country_year: int,
    base_path: str,
    file_template: str,
    alpha_scatter: float = 0.3,
    scatter_size: float = 2.0,
):

    plt.rcParams.update(
        {
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        }
    )

    fig, axes = plt.subplots(4, 2, figsize=(14, 24))
    cmap = plt.get_cmap("viridis", num_experiments)

    fig.subplots_adjust(left=0.08, right=0.98, hspace=0.3, wspace=0.2)

    for itr in tqdm(range(num_experiments)):
        try:
            model = torch.load(
                f"{base_path}/{file_template.format(itr = itr)}",
                weights_only=True,
            )
        except FileNotFoundError:
            print(f"FileNotFoundError in itr={itr}. Skipping.")
            continue

        color = cmap(itr)

        in_deg, out_deg = compute_degrees_from_model(model)
        _, edge_non_neg_weights, in_str, out_str = compute_strengths_from_model(model)
        edge_pos_weights = edge_non_neg_weights[edge_non_neg_weights > 0]
        pagerank = compute_pagerank(model)

        in_deg_pd = pd.Series(in_deg.detach().numpy()).round().astype(int)
        out_deg_pd = pd.Series(out_deg.detach().numpy()).round().astype(int)
        in_str_pd = pd.Series(in_str.detach().numpy() * monetary_unit)
        out_str_pd = pd.Series(out_str.detach().numpy() * monetary_unit)
        pagerank_pd = pd.Series(pagerank.detach().numpy())
        edge_pos_weights_pd = pd.Series(
            edge_pos_weights.detach().numpy() * monetary_unit
        )

        # CCDF: pagerank
        x, y = series_to_ccdf_dict(pagerank_pd)
        axes[0, 0].loglog(x, y, color=color, linewidth=0.8)

        # CCDF: edge weights
        x, y = series_to_ccdf_dict(edge_pos_weights_pd)
        axes[0, 1].loglog(x[x > 0], y[x > 0], color=color, linewidth=0.8)

        # CCDF: degrees
        x_in, y_in = series_to_ccdf_dict(in_deg_pd)
        x_out, y_out = series_to_ccdf_dict(out_deg_pd)
        axes[1, 0].loglog(
            x_in[x_in > 0], y_in[x_in > 0], color=color, linestyle="-.", linewidth=0.8
        )
        axes[1, 0].loglog(
            x_out[x_out > 0], y_out[x_out > 0], color=color, linewidth=0.8
        )

        # CCDF: strengths
        x_in_str, y_in_str = series_to_ccdf_dict(in_str_pd)
        x_out_str, y_out_str = series_to_ccdf_dict(out_str_pd)
        axes[1, 1].loglog(
            x_in_str[x_in_str > 0],
            y_in_str[x_in_str > 0],
            color=color,
            linestyle="-.",
            linewidth=0.8,
        )
        axes[1, 1].loglog(
            x_out_str[x_out_str > 0],
            y_out_str[x_out_str > 0],
            color=color,
            linewidth=0.8,
        )

        # Row 2, Col 0: out-degree vs out-strength
        axes[2, 0].scatter(
            x=out_deg_pd,
            y=out_str_pd,
            alpha=alpha_scatter,
            s=scatter_size,
            marker=".",
            color=color,
        )

        # Row 2, Col 1: in-degree vs in-strength
        axes[2, 1].scatter(
            x=in_deg_pd,
            y=in_str_pd,
            alpha=alpha_scatter,
            s=scatter_size,
            marker=".",
            color=color,
        )

        # Row 3, Col 0: out-degree vs in-degree
        axes[3, 0].scatter(
            x=out_deg_pd,
            y=in_deg_pd,
            alpha=alpha_scatter,
            s=scatter_size,
            marker=".",
            color=color,
        )

        # Row 3, Col 1: out-strength vs in-strength
        axes[3, 1].scatter(
            x=out_str_pd,
            y=in_str_pd,
            alpha=alpha_scatter,
            s=scatter_size,
            marker=".",
            color=color,
        )

    # formatting
    axes[0, 0].set_title("CCDF of pagerank")
    axes[0, 0].set_ylabel(r"$P(X \geq x)$")
    axes[0, 0].set_xlabel("pagerank")

    axes[0, 1].set_title("CCDF of edge weights")
    axes[0, 1].set_ylabel(r"$P(X \geq x)$")
    axes[0, 1].set_xlabel("weight (1 USD)")

    axes[1, 0].set_title("CCDF of in- and out-degree")
    axes[1, 0].set_ylabel(r"$P(X \geq x)$")
    axes[1, 0].set_xlabel("degree")
    legend_deg = [
        Line2D([0], [0], color=cmap(0), linestyle="-.", label="in-deg"),
        Line2D([0], [0], color=cmap(0), label="out-deg"),
    ]
    axes[1, 0].legend(handles=legend_deg, frameon=False)

    axes[1, 1].set_title("CCDF of in- and out-strength")
    axes[1, 1].set_ylabel(r"$P(X \geq x)$")
    axes[1, 1].set_xlabel("strength (1 USD)")
    legend_str = [
        Line2D([0], [0], color=cmap(0), linestyle="-.", label="in-str"),
        Line2D([0], [0], color=cmap(0), label="out-str"),
    ]
    axes[1, 1].legend(handles=legend_str, frameon=False)

    axes[2, 0].set_xscale("log")
    axes[2, 0].set_yscale("log")
    axes[2, 0].set_xlabel("out-degree")
    axes[2, 0].set_ylabel("out-strength (1 USD)")
    axes[2, 0].set_title("Out-degree vs out-strength")

    axes[2, 1].set_xscale("log")
    axes[2, 1].set_yscale("log")
    axes[2, 1].set_xlabel("in-degree")
    axes[2, 1].set_ylabel("in-strength (1 USD)")
    axes[2, 1].set_title("In-degree vs in-strength")

    axes[3, 0].set_xscale("log")
    axes[3, 0].set_yscale("log")
    axes[3, 0].set_xlabel("out-degree")
    axes[3, 0].set_ylabel("in-degree")
    axes[3, 0].set_title("Out-degree vs in-degree")

    axes[3, 1].set_xscale("log")
    axes[3, 1].set_yscale("log")
    axes[3, 1].set_xlabel("out-strength (1 USD)")
    axes[3, 1].set_ylabel("in-strength (1 USD)")
    axes[3, 1].set_title("Out-strength vs in-strength")

    # External panel labels
    label_style = dict(fontsize=18, fontweight="bold")

    axes[0, 0].text(-0.12, 1.08, "(A)", transform=axes[0, 0].transAxes, **label_style)
    axes[0, 1].text(-0.12, 1.08, "(B)", transform=axes[0, 1].transAxes, **label_style)
    axes[1, 0].text(-0.12, 1.08, "(C)", transform=axes[1, 0].transAxes, **label_style)
    axes[1, 1].text(-0.12, 1.08, "(D)", transform=axes[1, 1].transAxes, **label_style)
    axes[2, 0].text(-0.12, 1.08, "(E)", transform=axes[2, 0].transAxes, **label_style)

    plt.suptitle(
        f"Network diagnostics across {num_experiments} experiments for {country_code}{country_year} IOT and {num_firms} firms",
        fontsize=14,
        y=0.92,
    )

    # for the color bar at the bottom of the experiment
    norm = mpl.colors.Normalize(vmin=0, vmax=num_experiments - 1)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.02, pad=0.05)
    cbar.set_ticks(np.linspace(0, num_experiments - 1, 5))
    cbar.set_ticklabels([int(x) for x in np.linspace(0, num_experiments - 1, 5)])
    cbar.set_label("Color corresponding to the experiment index")

    plt.savefig(f"{base_path}/{num_experiments}_networks.png", bbox_inches="tight")
    for ax in fig.axes:
        ax.grid(False)
    plt.show()


def _resolve_metric_index(df: pd.DataFrame, prop: str) -> str:
    """Return the dataframe index label for a plotted property.

    Older compiled diagnostics CSVs use ``RMSE (100 MUSD)``; newer outputs from
    ``compute_network_stats`` use ``IOT RMSE (100 MUSD)``. This helper accepts
    either name so plotting code can stay unchanged.
    """
    if prop in df.index:
        return prop
    if prop.startswith("RMSE (") and not prop.startswith("IOT RMSE ("):
        iot_name = prop.replace("RMSE (", "IOT RMSE (", 1)
        if iot_name in df.index:
            return iot_name
    if prop.startswith("IOT RMSE ("):
        legacy_name = prop.replace("IOT RMSE (", "RMSE (", 1)
        if legacy_name in df.index:
            return legacy_name
    raise KeyError(prop)


def build_plot_df(dfs, labels, properties, group_col, latex: bool = False):
    if len(dfs) != len(labels):
        raise ValueError("`dfs` and `labels` must have the same length.")

    formatted_labels = labels.copy()
    if latex and group_col == "network_size":
        formatted_labels = [f"$N={label}$" for label in labels]

    rows = []

    for prop in properties:
        for df, label in zip(dfs, formatted_labels):
            # exclude the last two rows, which are the mean and std
            vals = df.loc[_resolve_metric_index(df, prop)].iloc[:-2].values

            for v in vals:
                rows.append({"property": prop, group_col: label, "value": v})

    return pd.DataFrame(rows)


def plot_selected_network_properties(
    *,
    dfs,
    labels,
    group_col,
    country_code=None,
    country_year=None,
    latex_labels: bool = False,
    palette=None,
    base_path: str = None,
):
    if len(dfs) != len(labels):
        raise ValueError("`dfs` and `labels` must have the same length.")

    mean_properties = ["Mean degree"]
    agg_properties = ["RMSE (100 MUSD)"]
    path_length_properties = ["Avg path length"]
    topology_properties = [
        "Degree assortativity",
        "Reciprocity",
        "Avg clustering",
        "Global clustering",
    ]
    hill_properties = [
        "Hill in-degree",
        "Hill out-degree",
        "Hill in-strength",
        "Hill out-strength",
        "Hill influence",
        "Hill weights",
    ]
    regression_slope_first_properties = ["OLS: so-ko", "OLS: si-ki"]
    regression_slope_second_properties = ["OLS: ko-so", "OLS: ki-si"]
    regression_other_properties = ["OLS: r2out", "OLS: r2in"]

    mean_df = build_plot_df(
        dfs=dfs,
        labels=labels,
        properties=mean_properties,
        group_col=group_col,
        latex=latex_labels,
    )
    agg_df = build_plot_df(
        dfs=dfs,
        labels=labels,
        properties=agg_properties,
        group_col=group_col,
        latex=latex_labels,
    )
    path_length_df = build_plot_df(
        dfs=dfs,
        labels=labels,
        properties=path_length_properties,
        group_col=group_col,
        latex=latex_labels,
    )
    topology_df = build_plot_df(
        dfs=dfs,
        labels=labels,
        properties=topology_properties,
        group_col=group_col,
        latex=latex_labels,
    )
    topology_df["property"] = topology_df["property"].replace(
        {
            "Degree assortativity": "Degree\nassortativity",
            "Avg clustering": "Average\nclustering",
            "Global clustering": "Global\nclustering",
        }
    )
    hill_df = build_plot_df(
        dfs=dfs,
        labels=labels,
        properties=hill_properties,
        group_col=group_col,
        latex=latex_labels,
    )
    hill_df["property"] = hill_df["property"].replace(
        {
            "Hill in-degree": "in\ndegree",
            "Hill out-degree": "out\ndegree",
            "Hill in-strength": "in\nstrength",
            "Hill out-strength": "out\nstrength",
            "Hill influence": "influence",
            "Hill weights": "weights",
        }
    )
    reg_slope_first_df = build_plot_df(
        dfs=dfs,
        labels=labels,
        properties=regression_slope_first_properties,
        group_col=group_col,
        latex=latex_labels,
    )
    reg_slope_first_df["property"] = reg_slope_first_df["property"].replace(
        {
            "OLS: so-ko": "out-strength\n~ out-degree",
            "OLS: si-ki": "in-strength\n~ in-degree",
        }
    )
    reg_slope_second_df = build_plot_df(
        dfs=dfs,
        labels=labels,
        properties=regression_slope_second_properties,
        group_col=group_col,
        latex=latex_labels,
    )
    reg_slope_second_df["property"] = reg_slope_second_df["property"].replace(
        {
            "OLS: ko-so": "out-degree\n~ out-strength",
            "OLS: ki-si": "in-degree\n~ in-strength",
        }
    )
    reg_other_df = build_plot_df(
        dfs=dfs,
        labels=labels,
        properties=regression_other_properties,
        group_col=group_col,
        latex=latex_labels,
    )
    reg_other_df["property"] = reg_other_df["property"].replace(
        {
            "OLS: r2out": "out",
            "OLS: r2in": "in",
        }
    )

    fig = plt.figure(figsize=(18, 24))
    gs = gridspec.GridSpec(3, 6, figure=fig, wspace=1.5, hspace=0.2)

    ax0 = fig.add_subplot(gs[0, 0:2])
    ax1 = fig.add_subplot(gs[0, 2:4])
    ax2 = fig.add_subplot(gs[0, 4:6])
    ax3 = fig.add_subplot(gs[1, 0:3])
    ax4 = fig.add_subplot(gs[1, 3:6])
    ax5 = fig.add_subplot(gs[2, 0:2])
    ax6 = fig.add_subplot(gs[2, 2:4])
    ax7 = fig.add_subplot(gs[2, 4:6])

    axes = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7]
    dfs_plot = [
        mean_df,
        agg_df,
        path_length_df,
        topology_df,
        hill_df,
        reg_slope_first_df,
        reg_slope_second_df,
        reg_other_df,
    ]
    ylims = [None, None, None, None, None, (0.7, 1.65), (0.2, 0.3), (0.175, 0.45)]

    if palette is None:
        palette = sns.color_palette(n_colors=len(labels))

    for ax, df_plot, ylim in zip(axes, dfs_plot, ylims):
        sns.boxplot(
            data=df_plot,
            x="property",
            y="value",
            hue=group_col,
            showfliers=False,
            width=0.4,
            palette=palette,
            ax=ax,
        )

        ax.set_xlabel("")
        ax.set_ylabel("Value", fontsize=20)
        ax.tick_params(axis="y", labelsize=20)
        ax.tick_params(axis="x", labelsize=20)

        if ylim is not None:
            ax.set_ylim(*ylim)

    ax0.set_ylabel("Mean degree")
    ax0.set_xlabel("")
    ax0.set_xticklabels([])

    ax1.set_ylabel("RMSE (100 MUSD)")
    ax1.set_xlabel("")
    ax1.set_xticklabels([])

    ax2.set_ylabel("Average path length")
    ax2.set_xlabel("")
    ax2.set_xticklabels([])

    ax4.set_ylabel("Hill exponent")
    ax4.set_xlabel("")
    ax4.tick_params(axis="x", labelsize=16)

    ax5.set_ylabel("OLS regression coefficient")
    ax5.set_xlabel("")
    ax5.tick_params(axis="x", labelsize=16)

    ax6.set_ylabel("OLS regression coefficient")
    ax6.set_xlabel("")
    ax6.tick_params(axis="x", labelsize=16)

    ax7.set_ylabel(r"Coefficient of determination, $R^2$")
    ax7.set_xlabel("")

    ax0.legend_.remove()
    ax1.legend_.remove()
    ax2.legend_.remove()
    ax3.legend_.remove()
    ax4.legend_.remove()
    ax5.legend_.remove()
    handles, legend_labels = ax4.get_legend_handles_labels()
    ax6.legend_.remove()
    ax7.legend_.remove()

    fig.legend(
        handles,
        legend_labels,
        title=group_col.replace("_", " ").title(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=len(labels),
        prop={"size": 20},
    )
    fig.legends[0].get_title().set_fontsize(20)

    if group_col == "network_size":
        title = f"Selected network properties for {country_code}{country_year} for different network sizes, with fixed mean degree 40"
        fig.suptitle(title, fontsize=20, y=0.9)
    elif group_col == "country":
        title = "Selected network properties across countries"
        fig.suptitle(title, fontsize=20, y=0.9)
    else:
        fig.suptitle("Selected network properties", fontsize=20, y=0.9)

    if base_path is not None:
        plt.savefig(f"{base_path}/selected_network_properties.png", bbox_inches="tight")

    for ax in fig.axes:
        ax.grid(False)
    plt.show()


def visualize_network_for_overleaf_si_twobyfour(
    num_experiments: int,
    num_firms: int,
    monetary_unit: float,
    country_code: str,
    country_year: int,
    base_path: str,
    file_template: str,
    alpha_scatter: float = 0.3,
    scatter_size: float = 2.0,
):
    """
    Visualize the network for beamer slides.
    """
    plt.rcParams.update(
        {
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        }
    )

    fig, axes = plt.subplots(2, 4, figsize=(28, 12))
    cmap = plt.get_cmap("viridis", num_experiments)

    fig.subplots_adjust(left=0.06, right=0.98, hspace=0.35, wspace=0.28)

    for itr in tqdm(range(num_experiments)):
        try:
            model = torch.load(
                f"{base_path}/{file_template.format(itr = itr)}",
                weights_only=True,
            )
        except FileNotFoundError:
            print(f"FileNotFoundError in itr={itr}. Skipping.")
            continue

        color = cmap(itr)

        in_deg, out_deg = compute_degrees_from_model(model)
        _, edge_non_neg_weights, in_str, out_str = compute_strengths_from_model(model)
        edge_pos_weights = edge_non_neg_weights[edge_non_neg_weights > 0]
        pagerank = compute_pagerank(model)

        in_deg_pd = pd.Series(in_deg.detach().numpy()).round().astype(int)
        out_deg_pd = pd.Series(out_deg.detach().numpy()).round().astype(int)
        in_str_pd = pd.Series(in_str.detach().numpy() * monetary_unit)
        out_str_pd = pd.Series(out_str.detach().numpy() * monetary_unit)
        pagerank_pd = pd.Series(pagerank.detach().numpy())
        edge_pos_weights_pd = pd.Series(
            edge_pos_weights.detach().numpy() * monetary_unit
        )

        # CCDF: pagerank
        x, y = series_to_ccdf_dict(pagerank_pd)
        axes[0, 0].loglog(x, y, color=color, linewidth=0.8)

        # CCDF: edge weights
        x, y = series_to_ccdf_dict(edge_pos_weights_pd)
        axes[0, 1].loglog(x[x > 0], y[x > 0], color=color, linewidth=0.8)

        # CCDF: degrees
        x_in, y_in = series_to_ccdf_dict(in_deg_pd)
        x_out, y_out = series_to_ccdf_dict(out_deg_pd)
        axes[0, 2].loglog(
            x_in[x_in > 0], y_in[x_in > 0], color=color, linestyle="-.", linewidth=0.8
        )
        axes[0, 2].loglog(
            x_out[x_out > 0], y_out[x_out > 0], color=color, linewidth=0.8
        )

        # CCDF: strengths
        x_in_str, y_in_str = series_to_ccdf_dict(in_str_pd)
        x_out_str, y_out_str = series_to_ccdf_dict(out_str_pd)
        axes[0, 3].loglog(
            x_in_str[x_in_str > 0],
            y_in_str[x_in_str > 0],
            color=color,
            linestyle="-.",
            linewidth=0.8,
        )
        axes[0, 3].loglog(
            x_out_str[x_out_str > 0],
            y_out_str[x_out_str > 0],
            color=color,
            linewidth=0.8,
        )

        # Row 1, Col 0: out-degree vs out-strength
        axes[1, 0].scatter(
            x=out_deg_pd,
            y=out_str_pd,
            alpha=alpha_scatter,
            s=scatter_size,
            marker=".",
            color=color,
        )

        # Row 1, Col 1: in-degree vs in-strength
        axes[1, 1].scatter(
            x=in_deg_pd,
            y=in_str_pd,
            alpha=alpha_scatter,
            s=scatter_size,
            marker=".",
            color=color,
        )

        # Row 1, Col 2: out-degree vs in-degree
        axes[1, 2].scatter(
            x=out_deg_pd,
            y=in_deg_pd,
            alpha=alpha_scatter,
            s=scatter_size,
            marker=".",
            color=color,
        )

        # Row 1, Col 3: out-strength vs in-strength
        axes[1, 3].scatter(
            x=out_str_pd,
            y=in_str_pd,
            alpha=alpha_scatter,
            s=scatter_size,
            marker=".",
            color=color,
        )

    # formatting
    axes[0, 0].set_title("CCDF of pagerank")
    axes[0, 0].set_ylabel(r"$P(X \geq x)$")
    axes[0, 0].set_xlabel("pagerank")

    axes[0, 1].set_title("CCDF of edge weights")
    axes[0, 1].set_ylabel(r"$P(X \geq x)$")
    axes[0, 1].set_xlabel("weight (1 USD)")

    axes[0, 2].set_title("CCDF of in- and out-degree")
    axes[0, 2].set_ylabel(r"$P(X \geq x)$")
    axes[0, 2].set_xlabel("degree")
    legend_deg = [
        Line2D([0], [0], color=cmap(0), linestyle="-.", label="in-deg"),
        Line2D([0], [0], color=cmap(0), label="out-deg"),
    ]
    axes[0, 2].legend(handles=legend_deg, frameon=False)

    axes[0, 3].set_title("CCDF of in- and out-strength")
    axes[0, 3].set_ylabel(r"$P(X \geq x)$")
    axes[0, 3].set_xlabel("strength (1 USD)")
    legend_str = [
        Line2D([0], [0], color=cmap(0), linestyle="-.", label="in-str"),
        Line2D([0], [0], color=cmap(0), label="out-str"),
    ]
    axes[0, 3].legend(handles=legend_str, frameon=False)

    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("out-degree")
    axes[1, 0].set_ylabel("out-strength (1 USD)")
    axes[1, 0].set_title("Out-degree vs out-strength")

    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("in-degree")
    axes[1, 1].set_ylabel("in-strength (1 USD)")
    axes[1, 1].set_title("In-degree vs in-strength")

    axes[1, 2].set_xscale("log")
    axes[1, 2].set_yscale("log")
    axes[1, 2].set_xlabel("out-degree")
    axes[1, 2].set_ylabel("in-degree")
    axes[1, 2].set_title("Out-degree vs in-degree")

    axes[1, 3].set_xscale("log")
    axes[1, 3].set_yscale("log")
    axes[1, 3].set_xlabel("out-strength (1 USD)")
    axes[1, 3].set_ylabel("in-strength (1 USD)")
    axes[1, 3].set_title("Out-strength vs in-strength")

    plt.suptitle(
        f"Network diagnostics across {num_experiments} experiments for {country_code}{country_year} IOT and {num_firms} firms",
        fontsize=14,
        y=0.92,
    )

    # for the color bar at the bottom of the experiment
    norm = mpl.colors.Normalize(vmin=0, vmax=num_experiments - 1)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.02, pad=0.05)
    cbar.set_ticks(np.linspace(0, num_experiments - 1, 5))
    cbar.set_ticklabels([int(x) for x in np.linspace(0, num_experiments - 1, 5)])
    cbar.set_label("Color corresponding to the experiment index")

    plt.savefig(
        f"{base_path}/{num_experiments}_networks_twobyfour.png", bbox_inches="tight"
    )
    for ax in fig.axes:
        ax.grid(False)
    plt.show()


def plot_selected_network_properties_twobyfour(
    *,
    dfs,
    labels,
    group_col,
    country_code=None,
    country_year=None,
    latex_labels: bool = False,
    palette=None,
    base_path: str = None,
):
    """
    Visualize the network for beamer slides.
    """
    if len(dfs) != len(labels):
        raise ValueError("`dfs` and `labels` must have the same length.")

    mean_properties = ["Mean degree"]
    agg_properties = ["RMSE (100 MUSD)"]
    path_length_properties = ["Avg path length"]
    topology_properties = [
        "Degree assortativity",
        "Reciprocity",
        "Avg clustering",
        "Global clustering",
    ]
    hill_properties = [
        "Hill in-degree",
        "Hill out-degree",
        "Hill in-strength",
        "Hill out-strength",
        "Hill influence",
        "Hill weights",
    ]
    regression_slope_first_properties = ["OLS: so-ko", "OLS: si-ki"]
    regression_slope_second_properties = ["OLS: ko-so", "OLS: ki-si"]
    regression_other_properties = ["OLS: r2out", "OLS: r2in"]

    mean_df = build_plot_df(
        dfs=dfs,
        labels=labels,
        properties=mean_properties,
        group_col=group_col,
        latex=latex_labels,
    )
    agg_df = build_plot_df(
        dfs=dfs,
        labels=labels,
        properties=agg_properties,
        group_col=group_col,
        latex=latex_labels,
    )
    path_length_df = build_plot_df(
        dfs=dfs,
        labels=labels,
        properties=path_length_properties,
        group_col=group_col,
        latex=latex_labels,
    )
    topology_df = build_plot_df(
        dfs=dfs,
        labels=labels,
        properties=topology_properties,
        group_col=group_col,
        latex=latex_labels,
    )
    topology_df["property"] = topology_df["property"].replace(
        {
            "Degree assortativity": "Degree\nassortativity",
            "Avg clustering": "Average\nclustering",
            "Global clustering": "Global\nclustering",
        }
    )
    hill_df = build_plot_df(
        dfs=dfs,
        labels=labels,
        properties=hill_properties,
        group_col=group_col,
        latex=latex_labels,
    )
    hill_df["property"] = hill_df["property"].replace(
        {
            "Hill in-degree": "in\ndegree",
            "Hill out-degree": "out\ndegree",
            "Hill in-strength": "in\nstrength",
            "Hill out-strength": "out\nstrength",
            "Hill influence": "influence",
            "Hill weights": "weights",
        }
    )
    reg_slope_first_df = build_plot_df(
        dfs=dfs,
        labels=labels,
        properties=regression_slope_first_properties,
        group_col=group_col,
        latex=latex_labels,
    )
    reg_slope_first_df["property"] = reg_slope_first_df["property"].replace(
        {
            "OLS: so-ko": "out-strength\n~ out-degree",
            "OLS: si-ki": "in-strength\n~ in-degree",
        }
    )
    reg_slope_second_df = build_plot_df(
        dfs=dfs,
        labels=labels,
        properties=regression_slope_second_properties,
        group_col=group_col,
        latex=latex_labels,
    )
    reg_slope_second_df["property"] = reg_slope_second_df["property"].replace(
        {
            "OLS: ko-so": "out-degree\n~ out-strength",
            "OLS: ki-si": "in-degree\n~ in-strength",
        }
    )
    reg_other_df = build_plot_df(
        dfs=dfs,
        labels=labels,
        properties=regression_other_properties,
        group_col=group_col,
        latex=latex_labels,
    )
    reg_other_df["property"] = reg_other_df["property"].replace(
        {
            "OLS: r2out": "out",
            "OLS: r2in": "in",
        }
    )

    fig = plt.figure(figsize=(32, 14))
    gs = gridspec.GridSpec(2, 4, figure=fig, wspace=0.35, hspace=0.38)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[0, 3])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    ax7 = fig.add_subplot(gs[1, 3])

    axes = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7]
    dfs_plot = [
        mean_df,
        agg_df,
        path_length_df,
        topology_df,
        hill_df,
        reg_slope_first_df,
        reg_slope_second_df,
        reg_other_df,
    ]
    ylims = [None, None, None, None, None, (0.7, 1.65), (0.2, 0.3), (0.175, 0.45)]

    if palette is None:
        palette = sns.color_palette(n_colors=len(labels))

    for ax, df_plot, ylim in zip(axes, dfs_plot, ylims):
        sns.boxplot(
            data=df_plot,
            x="property",
            y="value",
            hue=group_col,
            showfliers=False,
            width=0.4,
            palette=palette,
            ax=ax,
        )

        ax.set_xlabel("")
        ax.set_ylabel("Value", fontsize=20)
        ax.tick_params(axis="y", labelsize=20)
        ax.tick_params(axis="x", labelsize=20)

        if ylim is not None:
            ax.set_ylim(*ylim)

    ax0.set_ylabel("Mean degree")
    ax0.set_xlabel("")
    ax0.set_xticklabels([])

    ax1.set_ylabel("RMSE (100 MUSD)")
    ax1.set_xlabel("")
    ax1.set_xticklabels([])

    ax2.set_ylabel("Average path length")
    ax2.set_xlabel("")
    ax2.set_xticklabels([])

    ax4.set_ylabel("Hill exponent")
    ax4.set_xlabel("")
    ax4.tick_params(axis="x", labelsize=16)

    ax5.set_ylabel("OLS regression coefficient")
    ax5.set_xlabel("")
    ax5.tick_params(axis="x", labelsize=16)

    ax6.set_ylabel("OLS regression coefficient")
    ax6.set_xlabel("")
    ax6.tick_params(axis="x", labelsize=16)

    ax7.set_ylabel(r"Coefficient of determination, $R^2$")
    ax7.set_xlabel("")

    ax0.legend_.remove()
    ax1.legend_.remove()
    ax2.legend_.remove()
    ax3.legend_.remove()
    ax4.legend_.remove()
    ax5.legend_.remove()
    handles, legend_labels = ax4.get_legend_handles_labels()
    ax6.legend_.remove()
    ax7.legend_.remove()

    for ax in [ax3, ax4]:
        ax.tick_params(axis="x", labelsize=16, rotation=30)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")

    fig.legend(
        handles,
        legend_labels,
        title=group_col.replace("_", " ").title(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=len(labels),
        prop={"size": 20},
    )
    fig.legends[0].get_title().set_fontsize(20)

    plt.subplots_adjust(bottom=0.18)  # creates space for the legend

    if group_col == "network_size":
        title = f"Selected network properties for {country_code}{country_year} for different network sizes, with fixed mean degree 40"
        fig.suptitle(title, fontsize=20, y=0.95)
    elif group_col == "country":
        title = "Selected network properties across countries"
        fig.suptitle(title, fontsize=20, y=0.95)
    else:
        fig.suptitle("Selected network properties", fontsize=20, y=0.95)

    if base_path is not None:
        plt.savefig(
            f"{base_path}/selected_network_properties_twobyfour.png",
            bbox_inches="tight",
        )

    for ax in fig.axes:
        ax.grid(False)
    plt.show()
