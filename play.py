import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from gymnasium.wrappers import FrameStack
import numpy as np
import ale_py

# Load trained model
model = DQN.load("dqn_breakout_model_3") 

# Using the breakout environment
env = gym.make("ALE/Breakout-v5", render_mode="human")  

# Apply preprocessing wrappers
env = AtariWrapper(env)  
env = FrameStack(env, 4) 

# Reset the environment
obs, _ = env.reset()
obs = np.squeeze(obs) 

print("Observation shape after reset:", obs.shape)  # check the shape

# Play for 1000 steps
for step in range(1000):
    action, _states = model.predict(obs, deterministic=True)  
    obs, reward, done, truncated, info = env.step(action)  

    obs = np.squeeze(obs)  # Remove extra dimension after each step

    print(f"Step {step}: obs.shape={obs.shape}, reward={reward}")  

    env.render()  
    
    if done or truncated:
        obs, _ = env.reset()  
        obs = np.squeeze(obs)  # Again remove extra dimension if it resets
        print("Environment reset! New obs.shape:", obs.shape)

# Close the environment
env.close()
