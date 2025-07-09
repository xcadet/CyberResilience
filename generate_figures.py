from pathlib import Path

import matplotlib.pyplot as plt

from experimental_settings import OUTPUT_DIR as DATA_ROOT
from resilience_experiments import (
    generate_experimental_results,
    generate_experimental_results_lines,
    generate_experimental_results_cluster,
    generate_experimental_results_cumulative,
)
from scorer import ORIGINAL_SCORER, SCORER_V3, SCORER_V4
from weighter import ORIGINAL_WEIGHTER, WEIGHTER_V3, WEIGHTER_V4
from visualization import get_tab_colors

FIGURES_OUTPUT_DIR = Path("generated_figures")
FIGURES_OUTPUT_DIR.mkdir(exist_ok=True)
YLIM_LOWER = -0.05
YLIM_UPPER = 1.05

BLUE_MONITOR_COLOR = get_tab_colors(0)
BLUE_REACT_RESTORE_COLOR = get_tab_colors(1)
PPO_COLOR = get_tab_colors(2)
BLUE_R_COLOR = get_tab_colors(3)
BLUE_RD_COLOR = get_tab_colors(4)

AGENTS_COLOR_MARKER = [
    ("BlueMonitorAgent", BLUE_MONITOR_COLOR, "o"),
    ("BlueReactRestoreAgent", BLUE_REACT_RESTORE_COLOR, "s"),
    ("PPO", PPO_COLOR, "^"),
    ("Blue-R", BLUE_R_COLOR, "x"),
    ("Blue-D", BLUE_RD_COLOR, "v"),
]


def figure_2(data_root: Path = DATA_ROOT, output_dir: Path = FIGURES_OUTPUT_DIR):
    plt.rcdefaults()
    fig, ax = plt.subplots()
    fig.set_size_inches(4, 3)
    scenario = "SIS_2"
    eps_seed = 19
    scorer = ORIGINAL_SCORER
    weigher = ORIGINAL_WEIGHTER

    linestyle = [":", "-.", "--", "-"]
    legends = []
    handles = []  # List to store custom legend handles
    for window, color, style in zip(
        [1, 10, 50, 100],
        ["tab:green", "tab:orange", "tab:red", "tab:blue"],
        linestyle,
    ):
        (line,) = ax.plot(
            [], [], color=color, linestyle=style
        )  # Create a line for the legend
        handles.append(line)  # Add the line to the handles list
        generate_experimental_results(
            data_root,
            [("PPO", color, None, style)],
            [scenario],
            [0],
            [eps_seed],
            window,
            scorer,
            weigher,
            ax,
            save=False,
            use_lines=True,
        )
        legends.append(f"$\Delta t = {window}$")

    ax.set_title(None)
    ax.set_ylim(YLIM_LOWER, YLIM_UPPER)
    ax.legend(handles, legends)  # Use custom handles for the legend
    plt.tight_layout()
    save_path = output_dir / "figure_2.png"
    plt.savefig(save_path, dpi=300)
    print(f"Generated and saved Figure 2 ({save_path})")


def figure_3(data_root: Path = DATA_ROOT, output_dir: Path = FIGURES_OUTPUT_DIR):
    figure_3_a(data_root, output_dir)
    figure_3_b(data_root, output_dir)


def figure_3_a(data_root: Path = DATA_ROOT, output_dir: Path = FIGURES_OUTPUT_DIR):
    # use plotting script from the train directory
    pass


def figure_3_b(data_root: Path = DATA_ROOT, output_dir: Path = FIGURES_OUTPUT_DIR):
    SCENARIOS = [
        "SIS_2_fixed",
    ]
    fig, ax = plt.subplots()  # 1, 1, figsize=(10, 3))
    fig.set_size_inches(4, 3)

    for ax, scenario in zip([ax], SCENARIOS):
        generate_experimental_results_cumulative(
            Path("csv"),
            AGENTS_COLOR_MARKER,
            [scenario],
            [0],
            range(100),
            100,
            SCORER_V4,
            WEIGHTER_V4,
            ax,
            save=False,
        )
        ax.set_title(None)

    plt.tight_layout()
    save_path = output_dir / "figure_3b.png"
    plt.savefig(save_path, dpi=300)
    print(f"Generated and saved Figure 3b ({save_path})")


def figure_4(data_root: Path = DATA_ROOT, output_dir: Path = FIGURES_OUTPUT_DIR):
    plt.rcdefaults()
    _, axs = plt.subplots(1, 3, figsize=(10, 3))
    scenario = "SIS_2"
    eps_seed = 19

    axs_list = axs.flatten()
    generate_experimental_results(
        data_root,
        AGENTS_COLOR_MARKER,
        [scenario],
        [0],
        [eps_seed],
        100,
        ORIGINAL_SCORER,
        ORIGINAL_WEIGHTER,
        axs_list[0],
        save=False,
    )

    generate_experimental_results(
        data_root,
        AGENTS_COLOR_MARKER,
        [scenario],
        [0],
        [eps_seed],
        100,
        ORIGINAL_SCORER,
        WEIGHTER_V3,
        axs_list[1],
        save=False,
    )

    generate_experimental_results(
        data_root,
        AGENTS_COLOR_MARKER,
        [scenario],
        [0],
        [eps_seed],
        100,
        SCORER_V3,
        ORIGINAL_WEIGHTER,
        axs_list[2],
        save=False,
    )

    for ax in axs_list:
        ax.set_title(None)
        ax.set_ylim(YLIM_LOWER, YLIM_UPPER)

    plt.tight_layout()
    save_path = output_dir / "figure_4.png"
    plt.savefig(save_path, dpi=300)
    print(f"Generated and saved Figure 4 ({save_path})")


def figure_5(data_root: Path = DATA_ROOT, output_dir: Path = FIGURES_OUTPUT_DIR):
    PPO_ONLY = AGENTS_COLOR_MARKER[2]
    assert PPO_ONLY[0] == "PPO"
    path = data_root / "PPO" / "SIS_2" / "impact_types"
    agents = [PPO_ONLY]
    scenarios = ["SIS_2"]
    seeds = [0]
    episodes = range(100)
    window_size = 100
    scorer = ORIGINAL_SCORER
    weigher = ORIGINAL_WEIGHTER
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    ax = axs[0]
    generate_experimental_results_lines(
        path,
        agents,
        scenarios,
        seeds,
        episodes,
        window_size,
        scorer,
        weigher,
        ax=ax,
        save=False,
    )
    ax.set_title("Individual Attacks")

    ax = axs[1]
    generate_experimental_results_cluster(
        path,
        agents,
        scenarios,
        seeds,
        episodes,
        window_size,
        scorer,
        weigher,
        ax=ax,
        save=False,
    )
    ax.set_title("Clustered Attacks")
    ax = axs[2]
    generate_experimental_results(
        data_root,
        agents,
        scenarios,
        seeds,
        episodes,
        window_size,
        scorer,
        weigher,
        ax=ax,
        save=False,
    )
    ax.set_title("Averaged Attacks")
    for ax in axs:
        ax.axis([-0.5, 1100, -0.05, 1.05])

    fig.tight_layout()
    save_path = output_dir / "figure_5.png"
    plt.savefig(save_path, dpi=300)
    print(f"Generated and saved Figure 5 ({save_path})")


def figure_6(data_root: Path = DATA_ROOT, output_dir: Path = FIGURES_OUTPUT_DIR):
    seeds = range(1)
    episodes = range(100)
    window_size = 100

    combinations_scorer_weigher = [
        (ORIGINAL_SCORER, ORIGINAL_WEIGHTER),
        (ORIGINAL_SCORER, WEIGHTER_V3),
        (SCORER_V3, ORIGINAL_WEIGHTER),
    ]
    scenarios = ["SIS_1", "SIS_2_fixed", "SIS_3_fixed", "SIS_4", "SIS_5_fixed"]

    _, axs = plt.subplots(
        1,
        len(combinations_scorer_weigher),
        figsize=(4 * len(combinations_scorer_weigher), 3 * 1),
    )
    for (scorer, weigher), ax in zip(combinations_scorer_weigher, axs):
        generate_experimental_results(
            data_root,
            AGENTS_COLOR_MARKER,
            scenarios,
            seeds,
            episodes,
            window_size,
            scorer,
            weigher,
            ax=ax,
            save=False,
        )
        ax.axis([-0.5, 1100, -0.05, 1.05])
        ax.set_title("")
    plt.tight_layout()
    save_path = output_dir / "figure_6.png"
    plt.savefig(save_path, dpi=300)
    print(f"Generated and saved Figure 6 ({save_path})")


if __name__ == "__main__":
    figure_2()
    figure_3()
    figure_4()
    figure_5()
    figure_6()
