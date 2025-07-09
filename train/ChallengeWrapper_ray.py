from typing import Any
from gym import Env
from CybORG.Agents.Wrappers import BaseWrapper, OpenAIGymWrapper, BlueTableWrapper, RedTableWrapper, EnumActionWrapper, EnumActionWrapper2, TrueTableWrapper
from CybORG import CybORG
import inspect
from CybORG.Utility.utils import prettytable_to_dict, get_action_mapping
from datetime import datetime
import numpy as np


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


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self

    def step(self, action=None):
        self.obs, reward, done, info = self.env.step(action=action)

        self.step_counter += 1
        if self.max_steps is not None and self.step_counter >= self.max_steps:
            done = True

        terminated = done
        truncated = done

        self.obs = np.array(self.obs, dtype=np.float32)
        return self.obs, reward, terminated, truncated, info 


    def reset(self, *, seed=None, options=None):
        self.step_counter = 0
        self.obs = self.env.reset()
        self.obs = np.array(self.obs, dtype=np.float32)
        info = {}
        return self.obs, info

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
