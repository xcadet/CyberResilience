from pathlib import Path

from CybORG.Agents import BlueMonitorAgent, BlueReactRestoreAgent

from BlueStar_PPO_RLLIB import PPOAgent
from BlueStar_PPO_RLLIB_react_decoys import BlueStarDecoysAgent
from BlueStar_PPO_RLLIB_react_nodecoys import BlueStarReactAgent

SEEDS = range(30)
SCENARIOS = [
    "SIS",
    "SIS_1",
    "SIS_2",
    "SIS_2_fixed",
    "SIS_3",
    "SIS_3_fixed",
    "SIS_4",
    "SIS_5",
    "SIS_5_fixed",
]
NUM_EPISODES = 100
EPISODE_LENGTH = 1000
RANDOM_RED_INIT = False

OUTPUT_DIR = Path(f"csv")

AUTO_AGENT_SETTINGS_TYPE = 0
MANUAL_AGENT_SETTINGS_TYPE = 1


PPO_CHECKPOINT = "iter_235"
PPO_MODEL_PATH = Path(
    f"train/models/train_bline_100K_ppo_basic_avail/{PPO_CHECKPOINT}/policies/default_policy/"
)

BLUE_D_CHECKPOINT = "iter_235"
BLUE_D_MODEL_PATH = Path(
    f"train/models/train_bline_100K_react_decoys_allhosts/{BLUE_D_CHECKPOINT}/policies/default_policy/"
)

BLUE_R_CHECKPOINT = "iter_235"
BLUE_R_MODEL_PATH = Path(
    f"train/models/train_bline_100K_react_nodecoys/{BLUE_R_CHECKPOINT}/policies/default_policy/"
)

BLUE_AGENTS = {
    "Blue-R": (
        BlueStarReactAgent,
        MANUAL_AGENT_SETTINGS_TYPE,
        BLUE_R_CHECKPOINT,
        BLUE_R_MODEL_PATH,
    ),
    "Blue-D": (
        BlueStarDecoysAgent,
        MANUAL_AGENT_SETTINGS_TYPE,
        BLUE_D_CHECKPOINT,
        BLUE_D_MODEL_PATH,
    ),
    "PPO": (
        PPOAgent,
        MANUAL_AGENT_SETTINGS_TYPE,
        PPO_CHECKPOINT,
        PPO_MODEL_PATH,
    ),
    "BlueMonitorAgent": (
        BlueMonitorAgent,
        AUTO_AGENT_SETTINGS_TYPE,
        None,
        None,
    ),
    "BlueReactRestoreAgent": (
        BlueReactRestoreAgent,
        AUTO_AGENT_SETTINGS_TYPE,
        None,
        None,
    ),
}
