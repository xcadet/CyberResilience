import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# reads the tensorboard logs saved by rllib
tf_event_file = "logs/train_bline_100K_ppo_basic_avail/events.out.tfevents.1736263297"
TAG_NAME = "ray/tune/episode_reward_mean"
d1 = []
for e in tf.compat.v1.train.summary_iterator(tf_event_file):
    for v in e.summary.value:
        if v.tag == TAG_NAME:
            value = v.simple_value
            d1.append(value)
            
timestep_tag = "ray/tune/counters/num_env_steps_trained"
t1 = []
for e in tf.compat.v1.train.summary_iterator(tf_event_file):
    for v in e.summary.value:
        if v.tag == timestep_tag:
            value = v.simple_value
            t1.append(value)

tf_event_file = "logs/train_bline_100K_react_decoys_allhosts/events.out.tfevents.1737233496"
TAG_NAME = "ray/tune/episode_reward_mean"
d2 = []
for e in tf.compat.v1.train.summary_iterator(tf_event_file):
    for v in e.summary.value:
        if v.tag == TAG_NAME:
            value = v.simple_value
            d2.append(value)

timestep_tag = "ray/tune/counters/num_env_steps_trained"
t2 = []
for e in tf.compat.v1.train.summary_iterator(tf_event_file):
    for v in e.summary.value:
        if v.tag == timestep_tag:
            value = v.simple_value
            t2.append(value)            
            

tf_event_file = "logs/train_bline_100K_react_nodecoys/events.out.tfevents.1740160200"
TAG_NAME = "ray/tune/episode_reward_mean"
d3 = []
for e in tf.compat.v1.train.summary_iterator(tf_event_file):
    for v in e.summary.value:
        if v.tag == TAG_NAME:
            value = v.simple_value
            d3.append(value)

timestep_tag = "ray/tune/counters/num_env_steps_trained"
t3 = []
for e in tf.compat.v1.train.summary_iterator(tf_event_file):
    for v in e.summary.value:
        if v.tag == timestep_tag:
            value = v.simple_value
            t3.append(value)            

DATA = {"Blue-RD (reactive and proactive)": {"x": t2, "y": d2},
        "Blue-R (reactive)": {"x": t3, "y": d3},
        "PPO": {"x": t1, "y": d1}}

plt.figure(figsize=(4,3))
for key in DATA:
    plt.plot(DATA[key]["x"], DATA[key]["y"], label = key, linewidth=2.0)

plt.xlabel("Training Step") 
plt.ylabel("Reward") 
plt.legend()
plt.xlim((0, 25000000))
plt.ylim((-2000, -200))

plt.tight_layout()
plt.savefig("training_reward.png", dpi=300)


