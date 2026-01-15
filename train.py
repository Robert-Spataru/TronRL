import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback

# Import your environment
from tron_env import TronEnv

class TronSinglePlayerWrapper(gym.Env):
    """
    Wraps the Multi-Agent TronEnv to make it look like a Single-Agent Gymnasium Env.
    We fix Player 2 to be a Random Bot.
    """
    def __init__(self):
        super().__init__()
        self.env = TronEnv()
        
        # We only expose Player 1's spaces to the RL Agent
        self.observation_space = self.env.observation_space("player_1")
        self.action_space = self.env.action_space("player_1")
        
    def reset(self, seed=None, options=None):
        # 1. Reset the underlying multi-agent env
        obs_dict, info_dict = self.env.reset()
        
        # 2. Return only Player 1's observation
        return obs_dict["player_1"], info_dict["player_1"]

    def step(self, action):
        # 1. Create actions dictionary
        # AI controls Player 1
        # Random logic controls Player 2
        p2_action = self.env.action_space("player_2").sample()
        
        actions = {
            "player_1": action,
            "player_2": p2_action
        }
        
        # 2. Step the environment
        obs, rewards, terms, truncs, infos = self.env.step(actions)
        
        # 3. Extract Player 1's data
        # Note: If P1 dies, terms['player_1'] is True. 
        # If P2 dies, terms['player_1'] might be True (game over) or False depending on your logic.
        # Your env sets both to True if anyone dies, so this works.
        
        return (
            obs["player_1"], 
            rewards["player_1"], 
            terms["player_1"], 
            truncs["player_1"], 
            infos["player_1"]
        )

    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()

if __name__ == "__main__":
    # 1. Create Directories for logging
    models_dir = "models/PPO"
    log_dir = "logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 2. Instantiate and Check the Environment
    # We use the wrapper we just wrote
    env = TronSinglePlayerWrapper()
    
    # Simple check to ensure our wrapper follows gymnasium standards
    check_env(env) 
    print("Environment check passed!")

    # 3. Vectorize the Environment (Optional but recommended for speed)
    # This runs 4 games in parallel on your CPU
    # If using render(), use DummyVecEnv. If strictly training, SubprocVecEnv is faster.
    # vec_env = DummyVecEnv([lambda: TronSinglePlayerWrapper() for _ in range(4)])

    # 4. Initialize the Agent (PPO)
    # Policy: "CnnPolicy" because our observation is a Grid/Image (100x100x7)
    # Learning Rate: 0.0003 is standard, lower it if training is unstable
    model = PPO(
        "CnnPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=log_dir,
        learning_rate=0.0003,
        n_steps=2048,
    )

    # 5. Train
    print("Starting training...")
    TIMESTEPS = 100000 # Increase this to 1M or 5M for a smart bot
    model.learn(total_timesteps=TIMESTEPS)
    
    # 6. Save
    model.save(f"{models_dir}/tron_v1")
    print("Training Complete. Model Saved.")