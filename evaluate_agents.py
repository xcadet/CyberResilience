import argparse
import inspect
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import List

import gymnasium as gym
import numpy as np
import pandas as pd
import ray
from common import (
    format_file_name,
    get_num_cpus,
    get_scenario_based_on_topology,
    seed_everything,
    update_impact_count,
    update_impacts,
)
from CybORG import CybORG
from CybORG.Agents import B_lineAgent_SIS_random, GreenConsumeAgent
from CybORG.Agents.Wrappers import ChallengeWrapper, EnumActionWrapper2
from impact_reporters import ImpactReporter
from rich import print as print

from experimental_settings import (
    AUTO_AGENT_SETTINGS_TYPE,
    BLUE_AGENTS,
    EPISODE_LENGTH,
    MANUAL_AGENT_SETTINGS_TYPE,
    NUM_EPISODES,
    OUTPUT_DIR,
    RANDOM_RED_INIT,
    SCENARIOS,
)

sys.modules["gym"] = gym


def get_ray_temp_dir() -> str | None:
    """Get the ray temp directory.

    Returns:
        Path: The path to the ray temp directory.
    """
    maybe_path = os.getenv("RAY_TEMP_DIR")
    if maybe_path is not None:
        path = Path(maybe_path)
        path.mkdir(parents=True, exist_ok=True)
        return str(path)
    else:
        return None


def run_episode(
    env,
    cyborg,
    agent_name: str,
    blue_agent,
    blue_agent_settings_type: bool,
    scenario: str,
    output_dir: Path,
    random_topology: bool,
    num_steps: int,
    seed: int,
    episode_id: int,
):
    """
    Run an episode of the game.

    Args:
        env (gym.Env): The environment.
        cyborg (CybORG): The CybORG environment.
        agent_name (str): The name of the agent.
        blue_agent (Agent): The blue agent.
        blue_agent_settings_type (bool): Whether the blue agent is a manual agent.
        scenario (str): The scenario.
        output_dir (Path): The output directory.
        random_topology (bool): Whether the topology is random.
        num_steps (int): The number of steps in the episode.
        seed (int): The seed of the environment.
        episode_id (int): The episode seed.
    """

    # PREPARE THE METRICS
    total_reward = []
    impact_count = defaultdict(lambda: [])

    impact_reporter = ImpactReporter()

    impacts = defaultdict(lambda: 0)
    for host in ["Auth", "Database", "Front"]:
        impacts[host] = 0

    rewards = []
    impact_types: List[int] = []

    # RESET THE ENVIRONMENT
    episode_seed = episode_id
    observation = env.reset()
    seed_everything(episode_seed)

    # Blue agent needs to be aware of the new action space
    enum_action_wrapper = EnumActionWrapper2(env=cyborg, paddings=True)
    blue_action_space = enum_action_wrapper.get_action_space(agent="Blue")
    if blue_agent_settings_type == MANUAL_AGENT_SETTINGS_TYPE:
        blue_agent.set_initial_values(
            action_space=blue_action_space,
            observation=cyborg.get_observation(agent="Blue"),
        )

    for _ in range(num_steps):
        # Calculate this *BEFORE* calling cyborg.step()
        _ = cyborg.get_agent_state("True")

        if blue_agent_settings_type == MANUAL_AGENT_SETTINGS_TYPE:
            action = blue_agent.get_action(observation, action_space=blue_action_space)
            observation, reward, _, _ = env.step(action=action)
        else:
            cyborg.step()
            reward = cyborg.get_rewards()["Blue"]

        _ = cyborg.get_last_action("Blue")

        rewards.append(reward)

        red_action = cyborg.get_last_action("Red")
        red_success = cyborg.get_observation(agent="Red")["success"]
        impact_type, _ = impact_reporter(red_action, red_success)
        impact_types.append(impact_type)
        impacts = update_impacts(impacts, red_action, red_success)

    # datum = log_metrics(episode_id, sum(rewards), metrics)
    print(
        f"\nEpisode {episode_id}, reward = {sum(rewards)}, impact count = {sum(impacts.values())}"
    )

    # Computing running averages
    total_reward.append(sum(rewards))

    impact_count = update_impact_count(impact_count, impacts)

    impact_df_data = {"steps": np.arange(num_steps) + 1, "impact_type": impact_types}
    impact_df = pd.DataFrame(impact_df_data)
    output_path = (
        output_dir
        / "impact_types"
        / format_file_name(
            agent_name,
            scenario,
            random_topology,
            seed,
            episode_seed,
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    impact_df.to_csv(output_path, index=None)

    output_path = (
        output_dir
        / "stats"
        / format_file_name(
            agent_name,
            scenario,
            random_topology,
            episode_seed,
            episode_id,
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print("finished", scenario)


def main(args):
    # NOTE: We need to seed 3 times
    # 1. The global randomness (SEED @ 1)
    # 2. The Topology (SEED @ 2)
    # 3. The Episode (SEED @ 3)
    print(args)

    # STEP 1: Get the settings
    seed = args.seed
    seed_everything(seed=seed)  # SEED @ 1
    assert isinstance(seed, int)
    scenario = args.scenario
    random_topology = args.random_topology
    episode_ndx = args.episode_ndx
    episode_length = args.episode_length
    blue_agent_name = args.blue_agent_name
    output_dir = args.output_dir
    random_red_init = args.random_red_init

    assert blue_agent_name in BLUE_AGENTS
    blue_agent, blue_agent_settings_type, blue_checkpoint, blue_model = BLUE_AGENTS[
        blue_agent_name
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    # STEP 2: Choose the agents
    red_agent = B_lineAgent_SIS_random
    green_agent = GreenConsumeAgent
    cyborg_agents = {"Red": red_agent, "Green": green_agent}

    # If the agent can be handled automatically by cyborg, we add it to the
    # cyborg_agents
    if blue_agent_settings_type == AUTO_AGENT_SETTINGS_TYPE:
        cyborg_agents["Blue"] = blue_agent
        print("Adding Blue agent to cyborg_agents")
    print("CybORG agents:", cyborg_agents)

    if scenario is not None:
        scenario_name = (
            f"/Shared/Scenarios/SIS_Availability_Scenarios/Scenario2_{scenario}.yaml"
        )
        scenario_file = str(inspect.getfile(CybORG))[:-10] + scenario_name
        print(scenario_file)
    else:
        scenario_file = None

    # Get the scenario based on the topology
    # If random_topology is True, we use a random topology and scenario is None
    cyborg_scenario = get_scenario_based_on_topology(scenario_file, random_topology)
    if cyborg_scenario is None:
        assert random_topology, "Scenario is None, but random_topology is False"

    # NOTE: Do not move the cyborg/env outside the loop
    # Even if we call env.reset(), there seem to be no more
    # impacts
    cyborg = CybORG(
        cyborg_scenario,
        "sim",
        agents=cyborg_agents,
        random_red_init=random_red_init,
        random_topologies=random_topology,
        seed=seed,  # SEED @ 2
    )
    if blue_agent_settings_type == AUTO_AGENT_SETTINGS_TYPE:
        env = cyborg
    else:
        env = ChallengeWrapper(
            env=cyborg,
            agent_name="Blue",
            scenario_path=cyborg.scenario_file,
            paddings=True,
            max_steps=episode_length,
        )
    print(cyborg.scenario_file)

    # If the agent must be handled manually by cyborg we initialize it here
    agent_name = blue_agent_name
    if blue_agent_settings_type == MANUAL_AGENT_SETTINGS_TYPE:
        blue_agent_name += f"_{blue_checkpoint}"
        blue_agent = blue_agent(cyborg_env=cyborg, model=blue_model)

    print(
        f"blue={blue_agent.__class__.__name__}, red={red_agent.__name__}, green={green_agent.__name__}"
    )
    run_episode(
        env,
        cyborg,
        agent_name,
        blue_agent,
        blue_agent_settings_type,
        scenario if scenario is not None else "Random",
        output_dir,
        random_topology,
        episode_length,
        seed,
        episode_ndx,
    )


@ray.remote
def ray_main(args):
    return main(args)


def run_known_scenario_experiment(
    agent_name: str,
    scenario: str,
    seed: int,
    output_dir: Path,
    episode_ndx: int,
    episode_length: int,
    random_red_init: bool,
):
    args = argparse.Namespace(
        scenario=scenario,
        seed=seed,
        random_topology=False,
        output_dir=output_dir,
        episode_ndx=episode_ndx,
        episode_length=episode_length,
        blue_agent_name=agent_name,
        random_red_init=random_red_init,
    )
    return ray_main.remote(args)


def run_all_experiments():
    ray.init(
        _temp_dir=get_ray_temp_dir(),
        ignore_reinit_error=True,
        num_cpus=get_num_cpus(adjust=True),
        num_gpus=0,
    )
    to_get = []
    output_dir = OUTPUT_DIR

    for agent_name in BLUE_AGENTS:
        for scenario in SCENARIOS:
            # NOTE: For the fixed scenarios, the only seed that matters
            # is the episode_ seed
            seed = 0
            for episode_seed in range(NUM_EPISODES):
                to_get.append(
                    run_known_scenario_experiment(
                        agent_name,
                        scenario,
                        seed,
                        output_dir / agent_name / scenario,
                        episode_seed,
                        EPISODE_LENGTH,
                        RANDOM_RED_INIT,
                    )
                )

    print(f"Running {len(to_get)} tasks")

    ray.get(to_get)
    ray.shutdown()


if __name__ == "__main__":
    run_all_experiments()
