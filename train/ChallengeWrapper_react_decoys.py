from typing import Any
from gym import Env
from CybORG.Agents.Wrappers import BaseWrapper, OpenAIGymWrapper, BlueTableWrapper, RedTableWrapper, EnumActionWrapper, EnumActionWrapper2, TrueTableWrapper
from CybORG import CybORG
import inspect
from CybORG.Utility.utils import prettytable_to_dict, get_action_mapping
from datetime import datetime
import numpy as np
import copy
from gymnasium import Space, spaces
from CybORG.Shared.Actions import *
from agent_utils import *

USE_ALL_DECOYS = True # True = use all available decoys; False = enforce one decoys (more are optional)
ASSETS_ONLY = False  # True = only enforcing decoys on critical assets; False = enforce hardening all hosts

class ChallengeWrapper(Env, BaseWrapper):
    def __init__(
        self,
        agent_name: str,
        env,
        agent=None,
        reward_threshold=None,
        max_steps=None,
        scenario_path=None,
        paddings=False,
    ):
        super().__init__(env, agent, paddings)
        self.cyborg_env = env
        self.agent_name = agent_name
        self.paddings = paddings
        self.scenario_path = scenario_path
        # print(self.scenario_path)
        if agent_name.lower() == "red":
            table_wrapper = RedTableWrapper
        elif agent_name.lower() == "blue":
            table_wrapper = BlueTableWrapper
        else:
            raise ValueError("Invalid Agent Name")

        if self.scenario_path is None:
            raise ValueError("Empty Scenario path")
        
        env = table_wrapper(env, output_mode="vector", scenario_path=self.scenario_path, paddings=self.paddings)
        env = EnumActionWrapper(env, paddings=self.paddings)
        env = OpenAIGymWrapper(agent_name=agent_name, env=env, paddings=self.paddings)

        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_threshold = reward_threshold
        self.max_steps = max_steps
        self.step_counter = 0
        self.obs = None
        self.extend_obs_space()

        # hadling action space error
        self.reset()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self

    def step(self, action=None):
        self.obs, reward, done, info = self.env.step(action=action)
        
        self.step_counter += 1
        if self.max_steps is not None and self.step_counter >= self.max_steps:
            done = True

        terminated = done
        truncated = done
        
        self.decoy_history = update_decoy_history(self.blue_action_space, action, self.decoy_history_init, self.decoy_history)

        # construct action mask
        # if we know of any compromised hosts, select only among reactive actions like Remove, Restore

        user_hosts, escalated_hosts = get_compromised_hosts(self.cyborg_env, self.obs)

        if len(user_hosts) != 0 or len(escalated_hosts) != 0: # recover
            possible_reactions = get_possible_reactions(self.blue_action_space, [Restore])
            filtered_reactions = filter_reactions(user_hosts, escalated_hosts, possible_reactions, self.blue_action_space)
            action_mask = filter_action_mask(self.blue_action_space, filtered_reactions)
        else:
            indexes_to_deploy = []
            deployed, available = deployed_decoys(self.blue_action_space, self.decoy_history_init, self.decoy_history, assets_only=ASSETS_ONLY)
            #print(dict(deployed), dict(available))
            
            for host in available:
                if USE_ALL_DECOYS: # deploy all available decoys
                    indexes_to_deploy += available[host]
                elif len(deployed[host]) == 0:  # ensure at least one decoy per host
                    indexes_to_deploy += available[host]
           
            #print(indexes_to_deploy)
            if len(indexes_to_deploy) > 0: # harden
                action_mask = filter_action_mask(self.blue_action_space, indexes_to_deploy)
            else: # try any valid action
                action_mask = valid_action_mask(self.blue_action_space, self.decoy_history)


        self.obs = {"observations": self.obs,
                  "action_mask": action_mask,
        }

        return self.obs, reward, terminated, truncated, info 

    
    def reset(self, *, seed=None, options=None):
        self.step_counter = 0
        self.obs = self.env.reset()
        
        enum_w = EnumActionWrapper2(env=self.cyborg_env, paddings = True)
        self.blue_action_space = enum_w.get_action_space(agent='Blue')
       
        # initialize decoy history with valid decoys
        # valid means the correct OS and on ports that are not used by other processes
        decoy_history = reset_decoy_history(self.blue_action_space)
        observation = self.cyborg_env.get_observation(agent="Blue")
        self.decoy_history_init = prune_decoys_on_used_ports(self.cyborg_env, self.blue_action_space, observation, decoy_history)
        self.decoy_history = copy.deepcopy(self.decoy_history_init)

        action_mask = valid_action_mask(self.blue_action_space, self.decoy_history)
        
        self.obs = {"observations": self.obs,
            "action_mask": action_mask,
        }

        info = {}
        return self.obs, info
    

    def extend_obs_space(self):
        #print(self.action_space, self.action_space.n)
        
        self.observation_space = spaces.Dict({"observations": self.observation_space,
                        "action_mask": spaces.MultiDiscrete([2] * self.action_space.n),
                        })

    def get_attr(self, attribute: str):
        return self.env.get_attr(attribute)

    def get_observation(self, agent: str):
        return self.env.get_observation(agent)

    def get_agent_state(self, agent: str):
        return self.env.get_agent_state(agent)

    def get_action_space(self, agent=None) -> dict:
        return self.env.get_action_space(self.agent_name)

    def get_last_action(self, agent):
        return self.get_attr("get_last_action")(agent)

    def get_ip_map(self):
        return self.get_attr("get_ip_map")()

    def get_rewards(self):
        return self.get_attr("get_rewards")()

    def get_reward_breakdown(self, agent: str):
        return self.get_attr("get_reward_breakdown")(agent)

    def seed(self, seed: int):
        self.env.set_seed(seed)

    def get_action_mapping(self):
        enum_action_wrapper = EnumActionWrapper2(env=self.cyborg_env, paddings=self.paddings)
        action_space = enum_action_wrapper.get_action_space(agent=self.agent_name)

        # True Table
        true_table = TrueTableWrapper(self.cyborg_env)
        table = true_table.get_agent_state("True")
        true_table_dict = prettytable_to_dict(table)
        # print(table)

        # Agent Action Mapping
        action_mapping = {"status": "success"}
        for i in range(len(action_space)):
            action_dict = get_action_mapping(action_space[i], true_table_dict, i)
            action_mapping[i] = action_dict

        return action_mapping

    def get_encoded_action_from_action(self, action, agent_name):
        # Action Space is the same at every step
        enum_action_wrapper = EnumActionWrapper2(env=self.cyborg_env, paddings=self.paddings)
        action_space = enum_action_wrapper.get_action_space(agent=agent_name)

        action_space = [str(element) for element in action_space]
        num_action = action_space.index(str(action))
        return num_action
