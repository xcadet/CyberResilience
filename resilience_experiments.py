from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as mcolors

import matplotlib.pyplot as plt
import numpy as np
from clustering import agglomerative_clustering, reorder_clusters_by_size
from impact_reporters import AVAIL_DROP_KEY, CONF_DROP_KEY, INTEGRITY_DROP_KEY
from scipy.ndimage import gaussian_filter1d
from common import (
    cumulate_windows,
    get_grouped_weights,
    get_paths,
    get_paths_v2,
    get_scenario_max_value_per_window,
)
from visualization import plot_means_and_stds, prepare_ax

from scorer import ScorerChain

Y_AXIS_LABEL_DROP = "Resilience Drop"
Y_AXIS_LABEL = "Resilience"
X_AXIS_LABEL = "Steps"
LABEL_MAP = {
    "BlueReactRestoreAgent": "Restore",
    "BlueMonitorAgent": "Monitor",
    "PPO": "PPO",
    "Blue-D": "Blue-RD",
    "Blue-R": "Blue-R",
}

MIN_Y_AXIS_VALUE = -0.05
MAX_Y_AXIS_VALUE = 1.05


KEY_TO_YLABEL = {
    AVAIL_DROP_KEY: "Availability Drop",
    INTEGRITY_DROP_KEY: "Integrity Drop",
    CONF_DROP_KEY: "Confidentiality Drop",
}


def generate_plot(
    values: np.ndarray,
    windows_size: int,
    max_value_per_window: float,
    color: str | None = None,
    label: str | None = None,
    marker: str | None = None,
    ax: plt.Axes | None = None,
    linestyle: str = "-",
    ylabel: str = Y_AXIS_LABEL,
):
    generate_plot_using_gaussian_filter(
        values,
        windows_size,
        max_value_per_window,
        color,
        label,
        marker,
        ax,
        linestyle,
        ylabel,
    )


def compute_gaussian_filter(
    values: np.ndarray, windows_size: int, max_value_per_window: float
):
    sigma = windows_size / 2
    rolling_means = np.array(
        [gaussian_filter1d(row, sigma=sigma, mode="nearest") for row in values]
    )
    max_array_1d = np.full(len(rolling_means), max_value_per_window / windows_size)

    # Apply the 1D Gaussian filter
    filtered_array_1d = gaussian_filter1d(max_array_1d, sigma=sigma)

    # Get the maximum value after the convolution
    max_filtered_value_1d = np.max(filtered_array_1d)
    return rolling_means, max_filtered_value_1d


def generate_plot_using_gaussian_filter(
    values: np.ndarray,
    windows_size: int,
    max_value_per_window: float,
    color: str | None = None,
    label: str | None = None,
    marker: str | None = None,
    ax: plt.Axes | None = None,
    linestyle: str = "-",
    ylabel: str = Y_AXIS_LABEL,
):
    """Generate a plot of the resilience scores using a rolling mean.

    Args:
        values (np.ndarray): The resilience scores (matrix of observations).
        windows_size (int): The size of the window.
        max_value_per_window (float): The maximum value per window.
        color (str | None, optional): The color of the line. Defaults to None.
        label (str | None, optional): The label of the line. Defaults to None.
        marker (str | None, optional): The marker of the line. Defaults to None.
        ax (plt.Axes | None, optional): The axis to plot on. Defaults to None.
    """
    rolling_means, max_filtered_value_1d = compute_gaussian_filter(
        values, windows_size, max_value_per_window
    )
    rolling_means /= max_filtered_value_1d
    mean_rolling_mean = np.mean(rolling_means, axis=0)
    mean_rolling_mean = np.concatenate([np.zeros(1), mean_rolling_mean])
    std_rolling_mean = np.std(rolling_means, axis=0)
    std_rolling_mean = np.concatenate([np.zeros(1), std_rolling_mean])

    steps = np.arange(len(mean_rolling_mean))

    ax.plot(
        steps,
        mean_rolling_mean,
        color=color,
        label=LABEL_MAP[label],
        # marker=marker,
        linestyle=linestyle,
        linewidth=1,
        # markersize=0.1,
        alpha=1,
    )

    # Plot the standard deviation as a shaded area
    ax.fill_between(
        steps,
        (mean_rolling_mean - std_rolling_mean).clip(0)[: len(steps)],
        (mean_rolling_mean + std_rolling_mean)[: len(steps)],
        color=color,
        alpha=0.2,
    )
    ax.set_xlabel(X_AXIS_LABEL)
    ax.set_ylabel(ylabel)


def generate_plot_cumulative(
    values: np.ndarray,
    windows_size: int,
    max_value_per_window: float,
    color: str | None = None,
    label: str | None = None,
    marker: str | None = None,
    ax: plt.Axes | None = None,
    linestyle: str = "-",
    ylabel: str = Y_AXIS_LABEL,
):
    """Generate a plot of the resilience scores.

    Args:
        values (np.ndarray): The resilience scores.
        windows_size (int): The size of the window.
        max_value_per_window (float): The maximum value per window.
        color (str | None, optional): The color of the line. Defaults to None.
        label (str | None, optional): The label of the line. Defaults to None.
        marker (str | None, optional): The marker of the line. Defaults to None.
        ax (plt.Axes | None, optional): The axis to plot on. Defaults to None.
    """
    binned = cumulate_windows(values, windows_size)
    steps = np.arange(len(binned[0])) * windows_size
    plot_means_and_stds(
        ax=ax,
        values=binned,
        steps=steps,
        color=color,
        label=LABEL_MAP[label],
        marker=marker,
        linestyle=linestyle,
    )
    ax.set_xlabel(X_AXIS_LABEL)
    ax.set_ylabel("Total Red Impacts")


def generate_plot_individual_lines(
    values: np.ndarray,
    windows_size: int,
    max_value_per_window: float,
    label: str | None = None,
    marker: str | None = None,
    ax: plt.Axes | None = None,
    linestyle: str = "-",
    ylabel: str = "Resilience Score",
):
    generate_plot_individual_lines_by_sliding(
        values, windows_size, max_value_per_window, label, marker, ax, linestyle, ylabel
    )


def generate_plot_individual_lines_by_sliding(
    values: np.ndarray,
    windows_size: int,
    max_value_per_window: float,
    label: str | None = None,
    marker: str | None = None,
    ax: plt.Axes | None = None,
    linestyle: str = "-",
    ylabel: str = "Resilience Score",
):
    """Generate a plot of the resilience scores with lines colored using the viridis colormap.

    Args:
        values (np.ndarray): The resilience scores.
        windows_size (int): The size of the window.
        max_value_per_window (float): The maximum value per window.
        label (str | None, optional): The label of the line. Defaults to None.
        marker (str | None, optional): The marker of the line. Defaults to None.
        ax (plt.Axes | None, optional): The axis to plot on. Defaults to None.
        linestyle (str, optional): The line style. Defaults to "-".
        ylabel (str, optional): The label for the y-axis. Defaults to "Resilience Score".
    """
    if ax is None:
        _, ax = plt.subplots()

    # Bin the values
    rolling_means, max_filtered_value_1d = compute_gaussian_filter(
        values, windows_size, max_value_per_window
    )
    rolling_means /= max_filtered_value_1d
    # Get colormap and normalize
    num_lines, num_steps = rolling_means.shape
    cmap = cm.get_cmap("viridis", num_lines)
    norm = mcolors.Normalize(vmin=0, vmax=num_lines - 1)
    steps = np.arange(num_steps)

    # Plot each line with a different color from the colormap
    for i in range(num_lines):
        ax.plot(
            steps,
            rolling_means[i, :],
            color=cmap(norm(i)),
            marker=marker,
            linestyle=linestyle,
            alpha=0.5,
        )

    ax.set_xlabel(X_AXIS_LABEL)
    ax.set_ylabel(ylabel)

    if label:
        ax.legend()


def generate_experimental_results(
    root,
    methods_color_marker,
    scenarios,
    seeds,
    episodes,
    window_size,
    scorer,
    weigher,
    ax=None,
    save: bool = True,
    output_dir: Path = Path("paper_results"),
    use_lines: bool = False,
):
    ax = prepare_ax(ax)
    per_window_max = get_scenario_max_value_per_window(scorer, weigher, window_size)
    if not use_lines:
        for method, color, marker in methods_color_marker:
            paths = get_paths_v2(root, method, scenarios, seeds, episodes)
            weights = get_grouped_weights(paths, scorer, weigher)
            generate_plot(
                weights,
                window_size,
                per_window_max,
                color=color,
                label=method,
                marker=marker,
                ax=ax,
                ylabel=Y_AXIS_LABEL,
            )
    else:
        for method, color, marker, linestyle in methods_color_marker:
            paths = get_paths_v2(root, method, scenarios, seeds, episodes)
            weights = get_grouped_weights(paths, scorer, weigher)
            generate_plot(
                weights,
                window_size,
                per_window_max,
                color=color,
                label=method,
                marker=marker,
                linestyle=linestyle,
                ax=ax,
                ylabel=Y_AXIS_LABEL,
            )

    ax.legend()
    ax.set_title(
        f"{'-'.join(scenarios)} - {len(seeds)} seeds - {len(episodes)} episodes"
    )
    if save:
        plt.savefig(
            output_dir
            / f"{'-'.join(scenarios)}_{len(seeds)}_{len(episodes)}_episodes.png",
            dpi=300,
        )


def generate_experimental_results_cumulative(
    root,
    methods_color_marker,
    scenarios,
    seeds,
    episodes,
    window_size,
    scorer,
    weigher,
    ax=None,
    save: bool = True,
    output_dir: Path = Path("paper_results_jan24"),
):
    ax = prepare_ax(ax)
    per_window_max = get_scenario_max_value_per_window(scorer, weigher, window_size)
    for method, color, marker in methods_color_marker:
        paths = get_paths_v2(root, method, scenarios, seeds, episodes)
        paths = get_paths_v2(root, method, scenarios, seeds, episodes)
        weights = get_grouped_weights(paths, scorer, weigher)
        generate_plot_cumulative(
            weights,
            window_size,
            per_window_max,
            color=color,
            label=method,
            marker=marker,
            ax=ax,
            ylabel=Y_AXIS_LABEL,
        )
    ax.legend()
    ax.set_title(
        f"{'-'.join(scenarios)} - {len(seeds)} seeds - {len(episodes)} episodes"
    )
    if save:
        plt.savefig(
            output_dir
            / f"{'-'.join(scenarios)}_{len(seeds)}_{len(episodes)}_episodes.png",
            dpi=300,
        )


def generate_experimental_results_lines(
    root,
    methods_color_marker,
    scenarios,
    seeds,
    episodes,
    window_size,
    scorer,
    weigher,
    ax=None,
    save: bool = True,
    output_dir: Path = Path("paper_results"),
):
    ax = prepare_ax(ax)
    per_window_max = get_scenario_max_value_per_window(scorer, weigher, window_size)
    for method, color, marker in methods_color_marker:
        paths = get_paths(root, method, scenarios, seeds, episodes)
        weights = get_grouped_weights(paths, scorer, weigher)
        generate_plot_individual_lines(
            weights,
            window_size,
            per_window_max,
            ax=ax,
            ylabel=Y_AXIS_LABEL,
        )
    ax.set_title(
        f"{'-'.join(scenarios)} - {len(seeds)} seeds - {len(episodes)} episodes"
    )
    if save:
        plt.savefig(
            output_dir
            / f"{'-'.join(scenarios)}_{len(seeds)}_{len(episodes)}_episodes.png",
            dpi=300,
        )


def generate_plot_cluster(
    values: np.ndarray,
    windows_size: int,
    max_value_per_window: float,
    ax: plt.Axes | None = None,
    color: str | None = None,
    ylabel: str = Y_AXIS_LABEL,
):
    generate_plot_cluster_by_gaussian(
        values, windows_size, max_value_per_window, ax, color, ylabel
    )


def generate_plot_cluster_by_gaussian(
    values: np.ndarray,
    windows_size: int,
    max_value_per_window: float,
    ax: plt.Axes | None = None,
    color: str | None = None,
    ylabel: str = Y_AXIS_LABEL,
):
    """Generate a plot of the resilience scores.

    Args:
        values (_type_): _description_
        windows_size (int): _description_
        max_value_per_window (float): _description_
        color (_type_, optional): _description_. Defaults to None.
        label (_type_, optional): _description_. Defaults to None.
        marker (_type_, optional): _description_. Defaults to None.
        ax (_type_, optional): _description_. Defaults to None.
    """
    sigma = windows_size / 2
    rolling_means = np.array(
        [gaussian_filter1d(row, sigma=sigma, mode="nearest") for row in values]
    )
    max_array_1d = np.full(len(rolling_means), max_value_per_window / windows_size)

    # Apply the 1D Gaussian filter
    filtered_array_1d = gaussian_filter1d(max_array_1d, sigma=sigma)

    # Get the maximum value after the convolution
    max_filtered_value_1d = np.max(filtered_array_1d)

    num_experiments, num_steps = values.shape

    condensed = np.zeros((num_experiments, num_steps))
    for ndx in range(num_experiments):
        rolling_mean = rolling_means[ndx]
        condensed[ndx] = rolling_mean
    binned = condensed
    clusters = agglomerative_clustering(binned)
    # reorder cluster by size
    clusters = np.array(clusters)
    clusters = reorder_clusters_by_size(clusters)
    assert max(binned[0]) <= max_value_per_window
    binned /= max_filtered_value_1d
    assert max(binned[0]) <= 1
    i = 0
    for cluster, linestyle in zip(np.unique(clusters), ["-", "--", ":"]):
        indices = np.where(clusters == cluster)[0]
        cluster_size = len(indices)
        cluster_ratio = cluster_size / len(values)
        for_cluster = rolling_means[indices] / max_filtered_value_1d

        plot_means_and_stds(
            ax=ax,
            values=for_cluster,
            steps=np.arange(len(for_cluster[0])),
            color=color,
            label=f"Cluster {i} - {cluster_ratio * 100:.2f}%",
            # marker=marker,
            linestyle=linestyle,
        )
        i += 1
    assert min(binned[0]) >= MIN_Y_AXIS_VALUE
    assert max(binned[0]) <= MAX_Y_AXIS_VALUE
    ax.set_ylim(MIN_Y_AXIS_VALUE, MAX_Y_AXIS_VALUE)
    ax.set_xlabel(X_AXIS_LABEL)
    ax.set_ylabel(ylabel)


def generate_experimental_results_cluster(
    root: Path,
    methods_color_marker: list[tuple[str, str, str]],
    scenarios: list[str],
    seeds: list[int],
    episodes: list[int],
    window_size: int,
    scorer: ScorerChain,
    weigher,
    ax=None,
    save: bool = True,
    output_dir: Path = Path("paper_results"),
    use_color: bool = False,
):
    ax = prepare_ax(ax)
    per_window_max = get_scenario_max_value_per_window(scorer, weigher, window_size)
    for method, color, marker in methods_color_marker:
        paths = get_paths(root, method, scenarios, seeds, episodes)
        weights = get_grouped_weights(paths, scorer, weigher)
        generate_plot_cluster(
            weights,
            window_size,
            per_window_max,
            color=color if use_color else None,
            ax=ax,
        )
    ax.legend()
    ax.set_title(
        f"{'-'.join(scenarios)} - {len(seeds)} seeds - {len(episodes)} episodes"
    )
    if save:
        plt.savefig(
            output_dir
            / f"{'-'.join(scenarios)}_{len(seeds)}_{len(episodes)}_episodes_clusters.png",
            dpi=300,
        )
