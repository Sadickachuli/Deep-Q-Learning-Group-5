# Deep-Q-Learning-with-Atari-Group-5


![breakpic](https://github.com/user-attachments/assets/53d4d1c2-bd7b-4646-80ec-1dac49580b7a)

## 📌 Table of Contents

1. [Project Overview](#overview)
2. [Environment Setup](#environment)
3. [Training the DQN Agent](#training)
4. [Hyperparameter Tuning Results](#tuning)
5. [Running the Trained Agent](#running)
6. [Video Demonstration](#video)
7. [Key Takeaways](#takeaway)

<h2 id="overview"> 🎯 Project Overview</h2>

This project implements a Deep Q-Network (DQN) agent using Stable Baselines3 to play the Atari Breakout game. The agent learns through reinforcement learning by interacting with the environment and optimizing its policy to maximize rewards.

### Key Components

* ✅ train.py – Trains the DQN model and saves it.
* ✅ play.py – Loads the trained model and runs it in the environment.
* ✅ Hyperparameter Tuning – Tests different learning rates, discount factors, and exploration strategies.

<h2 id="environment"> ⚙️ Environment Setup</h2>

Dependencies

```sh
pip install stable-baselines3 gymnasium[atari] torch tensorboard
```

### Atari Environment

- Uses `BreakoutNoFrameskip-v4` (raw pixel observations).
- Preprocessed with frame stacking and grayscaling (handled by `Stable Baselines3`).

<h2 id="training"> 🤖 Training the DQN Agent</h2>


```sh
python train.py
```


### Key Training Parameters

|Parameter            |	Default Value |	Description                                          |
|---------------------|---------------|------------------------------------------------------|
|learning_rate	      |1e-4	          |Controls weight updates in the neural network.        |
|gamma                |	0.99          |	Discount factor for future rewards.                  |
|batch_size           |	32            |	Number of experiences sampled from replay buffer.    |
|epsilon_start        |	1.0           |	Initial exploration rate.                            |
|epsilon_end	        |0.01           |	Final exploration rate.                              |
|exploration_fraction	| 0.1           |	Decay rate of epsilon (exploration vs. exploitation).|


<h2 id="tuning"> 📊 Hyperparameter Tuning Results</h2>

We tested 4 different hyperparameter configurations and recorded their mean rewards over training.

|Learning Rate (lr) | Gamma (Y) | Batch size | Epsilon Start | Epsilon End | Exploration Fraction | Mean Reward |
|-------------------|-----------|------------|---------------|-------------|----------------------|-------------|
|1e-4               |0.99       |32          |1.0            |0.01         |0.1                   |59.8         |
|5e-4               |0.95       |64          |1.0            |0.05         |0.2                   |73.3         |
|2.5e-4             |0.97       |64          |1.0            |0.05         |0.15                  |77.6 (Best)  |
|3e-4               |0.98       |128         |1.0            |0.02         |0.12                  |52.5         |

### Analysis of Results

1. Best Performing Model (`lr=2.5e-4`, `gamma=0.97`, `batch_size=64`)
  - Achieved the highest mean reward (77.6).
  - Balanced exploration (`exploration_fraction=0.15`) and exploitation.
  - Moderate learning rate helped avoid instability.

2. High Learning Rate (`lr=5e-4`)
  - Faster learning but less stable (reward variance higher).
    
3. Large Batch Size (`batch_size=128`)
  - Lower reward (52.5) – Due to slower learning from fewer updates.
    
4. Default Parameters (`lr=1e-4`)
  - Stable but suboptimal compared to tuned versions.

<h2 id="running"> 🎮 Running the Trained Agent</h2>


```sh
python play.py
```

### Expected Output

- The agent plays 1000 episodes of Breakout.
- Real-time rendering of the game.
- Prints total reward per episode.
  
<h2 id="video"> Video Demonstration</h2>

https://github.com/user-attachments/assets/b26653e6-f7ca-4594-b6d1-884376f8e74c

## Model Comparison:" CNNPolicy vs. MLPPolicy

We compared CNNPolicy and MLPPolicy models using DQN to train an agent for Breakout:

- CNNPolicy outperformed the MLPPolicy, achieving an average reward of `59.8` and surviving `4940` timesteps per episode after `1 million` timesteps of training.

- MLPPolicy struggled, with an average reward of only `1.65` and `781` timesteps per episode after `500,000` timesteps.

Conclusion: The CNNPolicy is much more effective for visual tasks like Atari Breakout, as it processed the environment’s images better than the MLP model.

<h2 id="takeaway"> 🔑 Key Takeaways</h2>

1. Optimal Hyperparameters Matter
  - The best configuration (`lr=2.5e-4`, `gamma=0.97`) outperformed others.
  - Too high learning rates can destabilize training.
2. Exploration vs. Exploitation Trade-off
  - Lower `epsilon_end` (0.01) leads to more exploitation but may get stuck.
  - Higher `epsilon_end` (0.05) allows continued exploration.  
3. Batch Size Impact
  - Smaller batches (32-64) worked better than `128`.

<b>Due to the model size, it isn't uploadable to Git Hub. You can access them here: https://drive.google.com/drive/folders/1HMOAaSmMu2m-qnoQtI5VhU68KqUbTfCl?usp=sharing</b>



### Work Distribution
|Task                       | Assigned To             | Contribution Details                                         |
|---------------------------|-------------------------|--------------------------------------------------------------|
|Environment Setup          | Sadick                  | Installed dependencies, configured `BreakoutNoFrameskip-v4`  |
|DQN Training Script        | Sadick                  | Implemented `train.py` with callback logging                 |
|Hyperparameter Tuning      | Teniola                 | Tested 4 configurations, recorded mean rewards               |
|Play Script                | Muhammed                | Developed `play.py` with greedy policy                       |
|Video Documentation        | Sadick                  | Recorded agent gameplay, edited demo                         |
|README & Reporting         | Teniola                 | Wrote project documentation, results analysis                |


## 👥 Group Collaboration & Individual Contributions

Team Members:
1. [Sadick Mustapha](https://github.com/sadickachuli/)
2. [Teniola Ajani](https://github.com/elhameed/)
3. [Mohamed Yasin](https://github.com/mohamedAYasin/)

## 🚀 Conclusion

This project successfully trained a DQN agent to play Atari Breakout, with hyperparameter tuning improving performance. The best model achieved a mean reward of 77.6, demonstrating effective learning.

Future improvements:
- Implement Double DQN to reduce overestimation bias.
- Use frame stacking for better temporal awareness.

Enjoy your RL agent! 🎉



