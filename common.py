import random
import os
import numpy as np
import torch
from pathlib import Path
from statistics import mean
import pandas as pd
from typing import Dict
from CybORG.Shared.Enums import TrinaryEnum
from agent_utils import Impact
from scorer import ScorerChain
from weighter import Weighter

SEEDS = range(100)
SCENARIOS = ["SIS_1", "SIS_2", "SIS_3", "SIS_4", "SIS_5"]


def get_scenario_based_on_topology(scenario: str, random_topology: bool):
    if random_topology:
        return None
    else:
        return scenario


def get_num_cpus(adjust: bool = True) -> int:
    cpu_count = os.cpu_count()
    if adjust:
        return cpu_count - 1
    else:
        return cpu_count


def format_file_name(
    agent_name: str, scenario: str, random_topology: bool, seed: int, episode_id: int
):
    name = f"{agent_name}_{scenario if not random_topology else 'random_topology'}_seed{seed}"
    return f"{name}_episode{episode_id}.csv"


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_drops(
    avail_drop_cumulative: Dict,
    conf_drop_cumulative: Dict,
    integrity_drop_cumulative: Dict,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "step": list(avail_drop_cumulative.keys()),
            "avail_drop": list(avail_drop_cumulative.values()),
            "conf_drop": list(conf_drop_cumulative.values()),
            "integrity_drop": list(integrity_drop_cumulative.values()),
        }
    )
    return df


HOSTS = ["Auth", "Database", "Front"]


def update_impacts(impacts: Dict, red_action: Impact, red_success: TrinaryEnum) -> Dict:
    if type(red_action) == Impact and red_success == TrinaryEnum.TRUE:
        for host in HOSTS:
            if host in red_action.hostname:
                impacts[host] += 1
    return impacts


def update_impact_count(impact_count: Dict, impacts: Dict) -> Dict:
    for host in HOSTS:
        impact_count[host].append(impacts[host])
    return impact_count


def cumulate_windows(values: np.ndarray, window_size: int) -> np.ndarray:
    """Bin the values by window size.

    Args:
        values (np.ndarray): The values to bin. (#episode, #steps)
        window_size (int): The size of the window to bin by.

    Returns:
        np.ndarray: The binned values (#episode, #bins + 1).
            Where the first bin is always 0.
    """
    assert values.ndim == 2
    num_experiments, num_steps = values.shape
    num_bins = num_steps // window_size
    results = np.zeros((num_experiments, num_bins))
    for experiment_ndx in range(num_experiments):
        for bin_ndx in range(num_bins):
            start = bin_ndx * window_size
            end = start + window_size
            # start is 0
            results[experiment_ndx, bin_ndx] = np.sum(values[experiment_ndx, 0:end])
    # We add 0 before the first bin, so that the first bin is always 0
    zero_pre_bin = np.zeros((num_experiments, 1))
    results = np.concatenate([zero_pre_bin, results], axis=1)
    assert results.shape == (num_experiments, num_bins + 1)
    return results


def get_paths(
    dir: Path,
    agent_name: str,
    scenarios: list[str],
    seeds: list[int],
    episodes: list[int],
) -> list[str]:
    paths = []
    for scenario in scenarios:
        for seed in seeds:
            for episode in episodes:
                paths.append(
                    dir
                    / format_file_name(
                        agent_name,
                        scenario,
                        False,
                        seed,
                        episode,
                    )
                )
    return paths


def get_paths_v2(
    dir: Path,
    agent_name: str,
    scenarios: list[str],
    seeds: list[int],
    episodes: list[int],
) -> list[str]:
    paths = []
    for scenario in scenarios:
        for seed in seeds:
            for episode in episodes:
                paths.append(
                    dir
                    / agent_name
                    / scenario
                    / "impact_types"
                    / format_file_name(
                        agent_name,
                        scenario,
                        False,
                        seed,
                        episode,
                    )
                )
    return paths


def get_grouped_weights(paths, scorer, weighter) -> np.ndarray:
    """Get the grouped weights for the given paths, scorer and weighter.

    Args:
        paths (list[Path]): The paths to the csv files.
        scorer (ScorerChain): The scorer to use.
        weighter (Weighter): The weighter to use.

    Returns:
        np.ndarray: The grouped resilience scores.
    """
    impact_dfs = [pd.read_csv(path) for path in paths]
    scored = [scorer(impact_df) for impact_df in impact_dfs]
    weighted = [weighter(scored) for scored in scored]
    return np.stack([weighted["weighted"] for weighted in weighted])


def extract_max_score_per_column(original_scorer: ScorerChain) -> dict[str, float]:
    """Given the scorer return the maximal value per column.

    Args:
        original_scorer (ScorerChain): The scorer to use.

    Returns:
        dict[str, float]: A mapping from the column name to the maximal value.
    """
    name_to_max_value = {}
    for scorer in original_scorer.scorers:
        name_to_max_value[scorer.out_column] = max(scorer.mapping.values())
    return name_to_max_value


def get_scenario_max_value_per_window(
    scorer: ScorerChain, weighter: Weighter, window_size: int
) -> float:
    """Get the maximal value per window for the given scorer and weighter.

    Args:
        scorer (ScorerChain): The scorer to use.
        weighter (Weighter): The weighter to use.
        window_size (int): The window size to use.

    Returns:
        float: The maximal value per window.
    """
    column_to_max_value = extract_max_score_per_column(scorer)
    max_value_per_step = weighter.compute_max_value(column_to_max_value)
    per_window_max = max_value_per_step * window_size
    return per_window_max
