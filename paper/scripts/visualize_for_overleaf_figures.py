"""
Visualize the distributional properties of multiple networks, save them as .png files.
Also plot selected network properties as boxplots.

Plot styling:
    Only ``visualize_iot`` and ``visualize_network_for_overleaf_main`` use the
    scienceplots "science" aesthetic (via ``_with_science_style``). All other
    functions in this module use matplotlib's default style.
"""

from functools import wraps

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import gridspec
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from tqdm import tqdm

from network_generator.metrics import (
    compute_degrees_from_model,
    compute_pagerank,
    compute_strengths_from_model,
)
from network_generator.utils import firm_model_to_iot
from network_generator.visualization import series_to_ccdf_dict


def _with_science_style(fn):
    """Apply matplotlib's science style for a single plotting function."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        import scienceplots  # noqa: F401  (registers the "science" style)

        with plt.style.context(["science", "nature"]):
            return fn(*args, **kwargs)

    return wrapper


# The IOT matrices are expressed in units of 100 million USD.
IOT_UNIT_IN_USD = 1e8

# Reference blue: the first color of the "science" style cycle, i.e. the color
# of the single-line CCDFs (edge weights, pagerank). Reused for the in-curves
# and as the dark end of both heatmap colormaps so the whole figure is coherent.
COLOR_IN = "#0C5DA5"  # dark blue (solid in-curves)
COLOR_OUT = "#6BAED6"  # light blue (dashed out-curves)

# Sequential white -> blue colormap (matches the CCDF line color) for heatmaps.
WHITE_TO_BLUE_CMAP = LinearSegmentedColormap.from_list(
    "white_to_blue", ["white", COLOR_IN]
)

# Light-blue -> dark-blue colormap for the joint-density panels: starts from a
# visible light blue so low-but-nonzero density still stands out (empty bins
# stay white via the LogNorm "bad" color).
LIGHTBLUE_TO_BLUE_CMAP = LinearSegmentedColormap.from_list(
    "lightblue_to_blue", ["#d6e6f4", COLOR_IN]
)

# OECD ICIO 45-sector classification: (code, full name), ordered to match the
# rows/columns of the input-output table.
ICIO_SECTORS = (
    ("D01T02", "Agriculture, hunting, forestry"),
    ("D03", "Fishing and aquaculture"),
    ("D05T06", "Mining and quarrying, energy producing products"),
    ("D07T08", "Mining and quarrying, non-energy producing products"),
    ("D09", "Mining support service activities"),
    ("D10T12", "Food products, beverages and tobacco"),
    ("D13T15", "Textiles, textile products, leather and footwear"),
    ("D16", "Wood and products of wood and cork"),
    ("D17T18", "Paper products and printing"),
    ("D19", "Coke and refined petroleum products"),
    ("D20", "Chemical and chemical products"),
    ("D21", "Pharmaceuticals, medicinal chemical and botanical products"),
    ("D22", "Rubber and plastics products"),
    ("D23", "Other non-metallic mineral products"),
    ("D24", "Basic metals"),
    ("D25", "Fabricated metal products"),
    ("D26", "Computer, electronic and optical equipment"),
    ("D27", "Electrical equipment"),
    ("D28", "Machinery and equipment, nec"),
    ("D29", "Motor vehicles, trailers and semi-trailers"),
    ("D30", "Other transport equipment"),
    ("D31T33", "Manufacturing nec; repair and installation of machinery and equipment"),
    ("D35", "Electricity, gas, steam and air conditioning supply"),
    ("D36T39", "Water supply; sewerage, waste management and remediation activities"),
    ("D41T43", "Construction"),
    ("D45T47", "Wholesale and retail trade; repair of motor vehicles"),
    ("D49", "Land transport and transport via pipelines"),
    ("D50", "Water transport"),
    ("D51", "Air transport"),
    ("D52", "Warehousing and support activities for transportation"),
    ("D53", "Postal and courier activities"),
    ("D55T56", "Accommodation and food service activities"),
    ("D58T60", "Publishing, audiovisual and broadcasting activities"),
    ("D61", "Telecommunications"),
    ("D62T63", "IT and other information services"),
    ("D64T66", "Financial and insurance activities"),
    ("D68", "Real estate activities"),
    ("D69T75", "Professional, scientific and technical activities"),
    ("D77T82", "Administrative and support services"),
    ("D84", "Public administration and defence; compulsory social security"),
    ("D85", "Education"),
    ("D86T88", "Human health and social work activities"),
    ("D90T93", "Arts, entertainment and recreation"),
    ("D94T96", "Other service activities"),
    ("D97T98", "Activities of households as employers"),
)
ICIO_SECTOR_CODES = tuple(code for code, _ in ICIO_SECTORS)


def format_dollars(dollars: float, sig_figs: int = 3) -> str:
    """Format a USD amount with a K/M/B/T suffix, e.g. 1.1e8 -> "$110M".

    Values are rounded to ``sig_figs`` significant figures. The dollar sign is
    escaped when LaTeX rendering is active (a bare "$" would open math mode).
    """
    dollar = r"\$" if mpl.rcParams["text.usetex"] else "$"
    if dollars == 0:
        return f"{dollar}0"
    # Round to the requested number of significant figures.
    magnitude = np.floor(np.log10(abs(dollars)))
    scale = 10 ** (magnitude - (sig_figs - 1))
    dollars = np.round(dollars / scale) * scale
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(dollars) >= threshold:
            return f"{dollar}{dollars / threshold:.3g}{suffix}"
    return f"{dollar}{dollars:.3g}"


def format_usd_magnitude(value_in_iot_units: float, sig_figs: int = 1) -> str:
    """Format an IOT-unit value (100M USD) as a rounded dollar amount.

    Used for the IOT colorbar ticks, e.g. 0.21 -> "$20M".
    """
    return format_dollars(value_in_iot_units * IOT_UNIT_IN_USD, sig_figs=sig_figs)


def usd_axis_formatter(suppress_sub_cent: bool = True) -> mticker.FuncFormatter:
    """Tick formatter labelling USD-valued (log) axes as $1K / $1M / $1B.

    With ``suppress_sub_cent`` (the default), labels below $0.01 are hidden to
    avoid cluttering the low end with sub-cent values like $0.0001; pass False
    to show them.
    """

    def _fmt(value, _pos):
        if suppress_sub_cent and value != 0 and abs(value) < 0.005:
            return ""
        return format_dollars(value)

    return mticker.FuncFormatter(_fmt)


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


@_with_science_style
def visualize_iot(
    model: torch.Tensor,
    firm_to_industry_tensor: torch.Tensor,
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
    cmap = WHITE_TO_BLUE_CMAP
    norm = BoundaryNorm(quantiles, ncolors=cmap.N, clip=True)

    # Cells are drawn with pcolormesh (vector quads) rather than imshow so they
    # stay crisp at any zoom in the PDF; white cell borders form the grid.
    cell_edges = np.arange(num_industries + 1) - 0.5

    # Panel 1 (Row 0, Col 0): Heatmap of output aggregated IOT
    img1 = axes[0].pcolormesh(
        cell_edges,
        cell_edges,
        output_agg_iot_np,
        cmap=cmap,
        norm=norm,
        edgecolors="white",
        linewidth=0.4,
    )
    axes[0].invert_yaxis()
    axes[0].set_title(
        f"Aggregated synthetic network",
        fontsize=12,
    )

    # Panel 2 (Row 0, Col 1): Heatmap of true IOT
    img2 = axes[1].pcolormesh(
        cell_edges,
        cell_edges,
        data_iot_np,
        cmap=cmap,
        norm=norm,
        edgecolors="white",
        linewidth=0.4,
    )
    axes[1].invert_yaxis()
    axes[1].set_title(
        f"Empirical IOT ({str(country_code)} {str(country_year)})",
        fontsize=12,
    )

    # The Nature style uses inward (and minor) ticks, which protrude into the
    # heatmaps; drop all tick marks (keeping the labels).
    for ax in axes:
        ax.tick_params(which="both", length=0)
        ax.minorticks_off()

    # Label the axes with OECD ICIO sector codes (rows = supplier, cols = buyer).
    if len(ICIO_SECTOR_CODES) == num_industries:
        ticks = np.arange(num_industries)
        for ax in axes:
            ax.set_xticks(ticks)
            ax.set_xticklabels(
                ICIO_SECTOR_CODES,
                rotation=45,
                ha="right",
                rotation_mode="anchor",
                fontsize=5,
            )
            ax.set_yticks(ticks)
            ax.set_yticklabels(ICIO_SECTOR_CODES, fontsize=5)
            ax.set_xlabel("Buyer industry", fontsize=11)
        axes[0].set_ylabel("Supplier industry", fontsize=11)
    else:
        print(
            f"Warning: {len(ICIO_SECTOR_CODES)} sector codes do not match "
            f"{num_industries} industries; leaving numeric axis labels."
        )

    # Shared colorbar, with ticks at the quantile boundaries labelled as
    # dollar magnitudes (the matrices are in units of 100 million USD).
    cbar = fig.colorbar(
        img1,
        ax=[axes[0], axes[1]],
        orientation="horizontal",
        pad=0.12,
        fraction=0.05,
    )
    cbar.set_ticks(quantiles)
    cbar.set_ticklabels([format_usd_magnitude(q) for q in quantiles])
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


@_with_science_style
def visualize_network_for_overleaf_main(
    model: torch.Tensor,
    monetary_unit: float,
    save_path: str = None,
):
    """Visualize the network."""

    # ==========================================
    # Font size controls
    # ==========================================
    TITLE_FONTSIZE = 10
    LABEL_FONTSIZE = 9
    TICK_FONTSIZE = 8
    LEGEND_FONTSIZE = 8
    COLORBAR_LABEL_FONTSIZE = 8
    COLORBAR_TICK_FONTSIZE = 7
    PANEL_LABEL_FONTSIZE = 11
    CCDF_LINEWIDTH = 1.3

    # ==========================================
    # Colorbar layout controls
    # ==========================================
    COLORBAR_PAD = 0.08
    COLORBAR_FRACTION = 0.045

    # Panel labels are pinned top-left over a translucent box so they stay
    # legible even on top of the dense density panels.
    label_style = {
        "fontsize": PANEL_LABEL_FONTSIZE,
        "fontweight": "bold",
        "ha": "left",
        "va": "top",
        "bbox": dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.0),
    }

    # Two stacked subfigures so each row gets its own descriptor; the top row
    # holds the marginal (CCDF) panels, the bottom row the joint densities.
    fig = plt.figure(figsize=(9.5, 5.8), constrained_layout=True)
    subfig_top, subfig_bottom = fig.subfigures(2, 1, hspace=0.14)
    axes = np.array([subfig_top.subplots(1, 4), subfig_bottom.subplots(1, 4)])
    subfig_top.suptitle(
        "Marginal distributions", fontweight="bold", fontsize=TITLE_FONTSIZE + 1
    )
    subfig_bottom.suptitle(
        "Joint distributions", fontweight="bold", fontsize=TITLE_FONTSIZE + 1
    )

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
    axes[0, 0].loglog(
        x_in[x_in > 0],
        y_in[x_in > 0],
        color=COLOR_IN,
        linestyle="-",
        linewidth=CCDF_LINEWIDTH,
        label="In-degree",
    )
    axes[0, 0].loglog(
        x_out[x_out > 0],
        y_out[x_out > 0],
        color=COLOR_OUT,
        linestyle="--",
        linewidth=CCDF_LINEWIDTH,
        label="Out-degree",
    )
    axes[0, 0].set_title("CCDF degrees", fontsize=TITLE_FONTSIZE)
    axes[0, 0].set_xlabel("Degree", fontsize=LABEL_FONTSIZE)
    axes[0, 0].set_ylabel(r"$P(X\geq x)$", fontsize=LABEL_FONTSIZE)
    axes[0, 0].legend(fontsize=LEGEND_FONTSIZE, loc="lower left")

    # Row 0, Col 1: CCDF of strengths
    x_in_str, y_in_str = series_to_ccdf_dict(in_str_pd)
    x_out_str, y_out_str = series_to_ccdf_dict(out_str_pd)
    axes[0, 1].loglog(
        x_in_str[x_in_str > 0],
        y_in_str[x_in_str > 0],
        color=COLOR_IN,
        linestyle="-",
        linewidth=CCDF_LINEWIDTH,
        label="In-strength",
    )
    axes[0, 1].loglog(
        x_out_str[x_out_str > 0],
        y_out_str[x_out_str > 0],
        color=COLOR_OUT,
        linestyle="--",
        linewidth=CCDF_LINEWIDTH,
        label="Out-strength",
    )
    axes[0, 1].set_title("CCDF strengths", fontsize=TITLE_FONTSIZE)
    axes[0, 1].set_xlabel("Strength", fontsize=LABEL_FONTSIZE)
    axes[0, 1].xaxis.set_major_formatter(usd_axis_formatter(suppress_sub_cent=False))
    axes[0, 1].set_ylabel(r"$P(X\geq x)$", fontsize=LABEL_FONTSIZE)
    axes[0, 1].legend(fontsize=LEGEND_FONTSIZE, loc="lower left")

    # Row 0, Col 2: CCDF of edge weights
    x, y = series_to_ccdf_dict(edge_pos_weights_pd)
    axes[0, 2].loglog(x[x > 0], y[x > 0], linewidth=CCDF_LINEWIDTH)
    axes[0, 2].set_title("CCDF edge weights", fontsize=TITLE_FONTSIZE)
    axes[0, 2].set_xlabel("Edge weight", fontsize=LABEL_FONTSIZE)
    axes[0, 2].xaxis.set_major_formatter(usd_axis_formatter(suppress_sub_cent=False))
    axes[0, 2].set_ylabel(r"$P(X\geq x)$", fontsize=LABEL_FONTSIZE)

    # Row 0, Col 3: CCDF of pagerank
    x, y = series_to_ccdf_dict(pagerank_pd)
    axes[0, 3].loglog(x, y, linewidth=CCDF_LINEWIDTH)
    axes[0, 3].set_title("CCDF pagerank", fontsize=TITLE_FONTSIZE)
    axes[0, 3].set_xlabel("Pagerank", fontsize=LABEL_FONTSIZE)
    axes[0, 3].set_ylabel(r"$P(X\geq x)$", fontsize=LABEL_FONTSIZE)

    # Row 1, Col 0: out-degree vs out-strength
    h = axes[1, 0].hist2d(
        x=out_deg_pd,
        y=out_str_pd,
        bins=[
            np.logspace(np.log10(0.7), np.log10(1e5), 100),
            np.logspace(np.log10(1e-4), np.log10(1e11), 100),
        ],
        cmap=LIGHTBLUE_TO_BLUE_CMAP,
        norm=LogNorm(),
    )
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("Out-degree", fontsize=LABEL_FONTSIZE)
    axes[1, 0].set_ylabel("Out-strength", fontsize=LABEL_FONTSIZE)
    axes[1, 0].yaxis.set_major_formatter(usd_axis_formatter())

    cbar = subfig_bottom.colorbar(
        h[3],
        ax=axes[1, 0],
        location="bottom",
        orientation="horizontal",
        pad=COLORBAR_PAD,
        fraction=COLORBAR_FRACTION,
    )
    cbar.set_label("Counts", fontsize=COLORBAR_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE, which="both", length=0)

    # Row 1, Col 1: in-degree vs in-strength
    h = axes[1, 1].hist2d(
        x=in_deg_pd,
        y=in_str_pd,
        bins=[
            np.logspace(np.log10(0.7), np.log10(1e4), 100),
            np.logspace(np.log10(1e-6), np.log10(1e11), 100),
        ],
        cmap=LIGHTBLUE_TO_BLUE_CMAP,
        norm=LogNorm(),
    )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("In-degree", fontsize=LABEL_FONTSIZE)
    axes[1, 1].set_ylabel("In-strength", fontsize=LABEL_FONTSIZE)
    axes[1, 1].yaxis.set_major_formatter(usd_axis_formatter())

    cbar = subfig_bottom.colorbar(
        h[3],
        ax=axes[1, 1],
        location="bottom",
        orientation="horizontal",
        pad=COLORBAR_PAD,
        fraction=COLORBAR_FRACTION,
    )
    cbar.set_label("Counts", fontsize=COLORBAR_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE, which="both", length=0)

    # Row 1, Col 2: out-degree vs in-degree
    h = axes[1, 2].hist2d(
        x=out_deg_pd,
        y=in_deg_pd,
        bins=[
            np.logspace(np.log10(0.7), np.log10(1e5), 100),
            np.logspace(np.log10(0.7), np.log10(1e4), 100),
        ],
        cmap=LIGHTBLUE_TO_BLUE_CMAP,
        norm=LogNorm(),
    )
    axes[1, 2].set_xscale("log")
    axes[1, 2].set_yscale("log")
    axes[1, 2].set_xlabel("Out-degree", fontsize=LABEL_FONTSIZE)
    axes[1, 2].set_ylabel("In-degree", fontsize=LABEL_FONTSIZE)

    cbar = subfig_bottom.colorbar(
        h[3],
        ax=axes[1, 2],
        location="bottom",
        orientation="horizontal",
        pad=COLORBAR_PAD,
        fraction=COLORBAR_FRACTION,
    )
    cbar.set_label("Counts", fontsize=COLORBAR_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE, which="both", length=0)

    # Row 1, Col 3: out-strength vs in-strength
    h = axes[1, 3].hist2d(
        x=out_str_pd,
        y=in_str_pd,
        bins=[
            np.logspace(np.log10(1e-4), np.log10(1e11), 100),
            np.logspace(np.log10(1e-6), np.log10(1e11), 100),
        ],
        cmap=LIGHTBLUE_TO_BLUE_CMAP,
        norm=LogNorm(),
    )
    axes[1, 3].set_xscale("log")
    axes[1, 3].set_yscale("log")
    axes[1, 3].set_xlabel("Out-strength", fontsize=LABEL_FONTSIZE)
    axes[1, 3].set_ylabel("In-strength", fontsize=LABEL_FONTSIZE)
    axes[1, 3].xaxis.set_major_formatter(usd_axis_formatter())
    axes[1, 3].yaxis.set_major_formatter(usd_axis_formatter())

    cbar = subfig_bottom.colorbar(
        h[3],
        ax=axes[1, 3],
        location="bottom",
        orientation="horizontal",
        pad=COLORBAR_PAD,
        fraction=COLORBAR_FRACTION,
    )
    cbar.set_label("Counts", fontsize=COLORBAR_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE, which="both", length=0)

    # Panel labels
    panel_labels = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)"]

    # Pin every panel label to the top-left corner for fast scanning.
    for ax, label in zip(axes.flat, panel_labels):
        ax.text(0.03, 0.97, label, transform=ax.transAxes, **label_style)

    # Thin the major ticks to every two decades to cut clutter on the wide log
    # axes, and standardize tick label sizes.
    for ax in axes.flat:
        ax.xaxis.set_major_locator(mticker.LogLocator(base=100.0))
        ax.yaxis.set_major_locator(mticker.LogLocator(base=100.0))
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    # Light gridlines on the CCDF (marginal) panels help readers gauge slopes.
    for ax in axes[0]:
        ax.grid(True, which="major", linewidth=0.3, alpha=0.4)
    # Same light-gray major grid on the density (joint) panels, drawn on top of
    # the mesh so it stays visible.
    for ax in axes[1]:
        ax.set_axisbelow(False)
        ax.grid(True, which="major", linewidth=0.3, alpha=0.4, color="lightgray")

    # Thin divider at the seam between the two subfigures to separate the
    # marginal and joint rows.
    fig.add_artist(
        Line2D(
            [0.05, 0.95],
            [0.5, 0.5],
            transform=fig.transFigure,
            color="gray",
            linewidth=0.8,
        )
    )

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
