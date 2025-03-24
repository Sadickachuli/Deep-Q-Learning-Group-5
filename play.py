import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from gymnasium.wrappers import FrameStack
import numpy as np
import ale_py

# Load the trained model
model = DQN.load("dqn_breakout_model")  # Ensure this is the correct path to your trained model

# Create the environment
env = gym.make("ALE/Breakout-v5", render_mode="human")  # Human mode for visualization

# Apply preprocessing wrappers
env = AtariWrapper(env)  # Resize to (84, 84)
env = FrameStack(env, 4)  # Stack last 4 frames to match model input

# Reset the environment
obs, _ = env.reset()
obs = np.squeeze(obs)  # 🔧 Fix: Remove extra dimension (4, 84, 84, 1) -> (4, 84, 84)

print("Observation shape after reset:", obs.shape)  # Debugging print

# Play for 1000 steps
for step in range(1000):
    action, _states = model.predict(obs, deterministic=True)  # Get action from model
    obs, reward, done, truncated, info = env.step(action)  # Step in environment

    obs = np.squeeze(obs)  # 🔧 Fix: Remove extra dimension after each step

    print(f"Step {step}: obs.shape={obs.shape}, reward={reward}")  # Debugging print

    env.render()  # Render game
    
    if done or truncated:
        obs, _ = env.reset()  # Reset environment if game ends
        obs = np.squeeze(obs)  # 🔧 Fix: Remove extra dimension
        print("Environment reset! New obs.shape:", obs.shape)

# Close the environment
env.close()
