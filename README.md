# Quantitative Resilience Modeling for Autonomous Cyber Defense

This repository is the official implementation of [Quantitative Resilience Modeling for Autonomous
Cyber Defense](https://arxiv.org/abs/2503.02780)


## Training

The agents developed in this work have been trained and evaluated on an extension of the [CybORG CAGE 2 framework](https://github.com/cage-challenge/cage-challenge-2), which supports more complex network topologies but uses the same actions and observations.

Training requires ray 2.10.0. 
The agents presented in the paper are standard PPO, Blue-R (reactive), and Blue-RD (reactive + proactive decoys). 
To train each of these agents, run the corresponding script from the `train` directory:

```train
python training_ppo.py 
python training_react.py 
python training_react_decoys.py
```

Alternatively, we provide the model checkpoints in `train/models`


## Evaluation
The [experimental settings](./experimental_settings.py) determine which agents, scenarios, and seeds are used.

Based on these values, we evaluate the agents:
```eval
python evaluate_agents.py
```
Which generates the observed impacts for each attack.

Alternatively, we provide `csv.zip,` which contains the recorded impacts for the different agents.

## To generate the figures:
To generate figure 2, 3b, 4, 5, 6:
```figures
python generate_figures.py
```
To generate figure 3a:
```
cd train
python plot_training.py
```
