from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors


def get_n_colors(k: int, colormap_name: str = "viridis") -> list:
    """Returns a list of k colors from the specified colormap.

    Args:
        k: Number of colors to generate
        colormap_name: Name of the matplotlib colormap to use (default: 'viridis')

    Returns:
        List of k colors as RGB tuples
    """
    # Get the colormap
    color_map = plt.get_cmap(colormap_name)

    # Generate k evenly spaced numbers between 0 and 1
    colors = [color_map(i / (k - 1)) if k > 1 else color_map(0) for i in range(k)]

    return colors


def get_tab_colors(ndx):
    names = list(mcolors.TABLEAU_COLORS.keys())
    return mcolors.TABLEAU_COLORS[names[ndx]]


def plot_means_and_stds(
    ax,
    values,
    steps,
    color=None,
    linewidth=None,
    label=None,
    marker=None,
    linestyle="-",
):
    means_per_steps = np.mean(values, axis=0)
    stds_per_steps = np.std(values, axis=0)
    (line,) = ax.plot(
        steps,
        means_per_steps,
        color=color,
        alpha=1,
        marker=marker,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
    )
    auto_color = line.get_color()
    ax.fill_between(
        steps,
        (means_per_steps - stds_per_steps).clip(0),
        means_per_steps + stds_per_steps,
        color=auto_color,
        alpha=0.2,
        linestyle=linestyle,
    )


def plot_violin_with_means(
    ax, values, steps, color=None, width=25, linewidth=None, label=None
):
    """Creates a violin plot with overlaid mean values on the given axis.

    Args:
        ax: Matplotlib axis object to plot on
        values: List of data points for each step to create violin plots
        steps: List/array of x-axis positions for each violin
        means_per_steps: List/array of mean values to overlay on the plot
        width: Width of the violin plots (default: 25)
    """
    # Create violin plot with means shown
    means_per_steps = np.mean(values, axis=0)
    violin_parts = ax.violinplot(
        values,
        positions=steps,
        widths=width,
        showmeans=True,
    )
    # if color is no it should mathc the violinplot colors
    color = violin_parts["bodies"][0].get_facecolor().flatten()

    # Overlay mean values as a black line with markers
    ax.plot(
        steps,
        means_per_steps,
        color=color,
        alpha=0.5,
        marker="o",
        linewidth=linewidth,
        label=label,
    )


import pandas as pd


def visualize_method(dfs: List[pd.DataFrame]):
    metrics_to_values = {}

    _, axs = plt.subplots(4, 3, figsize=(3 * 10, 10))
    # _, axs = plt.subplots(2, 3, figsize=(3 * 10, 10))
    for plot_ndx, metric in enumerate(["avail_drop", "conf_drop", "integrity_drop"]):
        # We are going to compute once, assuming we have N seeds, and M steps
        # We want to build a matrix of N x M, where each cell is the value of the metric for the given seed and step
        # steps = dfs[0]["step"].unique()
        steps = dfs[0]["step"]
        values = np.zeros((len(dfs), len(steps)))
        # We now need to fill the matrix
        for ndx, df in enumerate(dfs):
            values[ndx] = df[metric]
        metrics_to_values[metric] = values
        # We now need to plot the matrix
        for ndx, color in enumerate(get_n_colors(len(dfs))):
            row = values[ndx]
            ax = axs[0, plot_ndx]
            ax.scatter(steps, row, color=color, alpha=0.5)
            ax = axs[1, plot_ndx]
            ax.plot(steps, row, color=color, alpha=0.5, marker="o")
        ax = axs[2, plot_ndx]
        plot_means_and_stds(ax, values, steps, linewidth=1)
        means_per_steps = np.mean(values, axis=0)
        # stds_per_steps = np.std(values, axis=0)
        # ax.plot(steps, means_per_steps, color="black", alpha=0.5, marker="o")
        # ax.fill_between(steps, means_per_steps - stds_per_steps, means_per_steps + stds_per_steps, color="black", alpha=0.2)
        # Now we want to do boxplots with the mean going through the center of the box
        ax = axs[3, plot_ndx]
        # ax.boxplot(values, positions=steps, widths=25)
        # ax.violinplot(values, positions=steps, widths=25, showmeans=True)
        # ax.plot(steps, means_per_steps, color="black", alpha=0.5, marker="o")
        plot_violin_with_means(ax, values, steps, means_per_steps)
        # I want to guarantee that the y-axis is between 0 and 1000, and shows all the way to 1000
        for ax in axs[:, plot_ndx]:
            ax.set_ylim(0, 1000)
            ax.set_yticks(np.arange(0, 1000, 100))
        ax.set_title(scenario_name + f" - {metric}")
        # ax.legend()
    plt.tight_layout()
    plt.show()
    return metrics_to_values


def prepare_ax(ax: plt.Axes | None = None, figsize: tuple[int, int] = (10, 3)):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    return ax
