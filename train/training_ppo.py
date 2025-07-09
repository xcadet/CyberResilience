
import gymnasium as gym
import inspect
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import torch as th

from CybORG import CybORG

from ChallengeWrapper_ray import ChallengeWrapper
from CybORG.Agents.Wrappers import EnumActionWrapper, OpenAIGymWrapper

from CybORG.Agents import *

import warnings
warnings.filterwarnings("ignore")

from ray.rllib.algorithms.ppo import PPOConfig, PPO, PPOTorchPolicy
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils import check_env
from ray.tune import register_env

# 2. Choose red agent:
#red_agent = RedMeanderAgent_SIS_random
red_agent = B_lineAgent_SIS_random

# 3. Randomness:
random_topology = True
random_red_init = True
topology_seed = None

max_steps = 500

def env_creator_CC2(env_config: dict):
    cyborg = CybORG(None, 'sim', agents={'Red': red_agent, 'Green': GreenConsumeAgent}, random_red_init=random_red_init, random_topologies=random_topology, seed = topology_seed)
    env = ChallengeWrapper(env=cyborg, agent_name="Blue", max_steps = max_steps, scenario_path=cyborg.scenario_file, paddings=True)
    return env

register_env(name="CC2", env_creator=lambda config: env_creator_CC2(config))
env = env_creator_CC2({})

algo_config = (
    PPOConfig()
    .framework("torch")
    .debugging(logger_config={"logdir":"logs/train_bline_100K_ppo_basic_avail", "type":"ray.tune.logger.TBXLogger"})
    .environment(env="CC2", disable_env_checking=True)
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    #.resources(num_gpus=1)  # export CUDA_VISIBLE_DEVICES=0,1
    .experimental(
        _disable_preprocessor_api=True,
    )
    .rollouts(
        batch_mode="complete_episodes",
        num_rollout_workers=30, # for debugging, set this to 0 to run in the main thread
    )
    .training(
        sgd_minibatch_size=32768, # default 128
        train_batch_size=100000, # default 4000
    )
)

model_dir = "models/train_bline_100K_ppo_basic_avail"

algo = algo_config.build()

start_time = time.time()

for i in range(1000):
    train_info = algo.train()
    print("\nIteration:", i, train_info)
    model_dir_crt = os.path.join(model_dir, "iter_"+str(i))
    print("\nSaving model in:", model_dir_crt)
    algo.save(model_dir_crt)

algo.save(model_dir_crt)

end_time = time.time()
print(f"Training time: {'{0:.2f}'.format(end_time - start_time)}")

