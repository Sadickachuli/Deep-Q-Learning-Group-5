# -*- coding: utf-8 -*-
"""MLPPtraining.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/14QPDzSZFipT6rUfErg4cxVdKWOXb152E
"""

!pip install stable-baselines3==2.1.0 gymnasium[atari] ale-py imageio matplotlib

!pip install gymnasium[accept-rom-license]

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

env_id = "BreakoutNoFrameskip-v4"

# Create Atari environment with 4 parallel instances
env = make_atari_env(env_id, n_envs=4, seed=42)
env = VecFrameStack(env, n_stack=4)

# Use MLP policy
policy = "MlpPolicy"

model = DQN(
    policy,
    env,
    learning_rate=1e-3,  # Faster learning
    gamma=0.99,
    batch_size=64,  # Process more samples per step
    buffer_size=10_000,  # Smaller buffer
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,  # Reduce exploration decay speed
    exploration_fraction=0.2,
    target_update_interval=500,  # Update more frequently
    train_freq=1,  # Train more often
    gradient_steps=2,  # More gradient updates per step
    verbose=1,
    tensorboard_log="./dqn_tensorboard/"
)

# Configure logging
logger = configure("./dqn_logs_mlp/", ["stdout", "csv", "tensorboard"])
model.set_logger(logger)

# Checkpoint callback
checkpoint_callback = CheckpointCallback(save_freq=5000, save_path="./checkpoints_mlp/")

# Train for 500,000 steps
model.learn(
    total_timesteps=500_000,
    callback=checkpoint_callback,
    tb_log_name="dqn_breakout_mlp_fast"
)

# Save the trained model
model.save("dqn_breakout_mlp_fast")

env.close()