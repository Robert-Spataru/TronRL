import time
from stable_baselines3 import PPO
from train import TronSinglePlayerWrapper # Import the wrapper from your train script

# 1. Load Environment
env = TronSinglePlayerWrapper()

# 2. Load Model
model_path = "models/PPO/tron_v1.zip"
model = PPO.load(model_path, env=env)

# 3. Play Loop
obs, _ = env.reset()
done = False

while not done:
    # Predict the best action (deterministic=True creates more stable behavior)
    action, _states = model.predict(obs, deterministic=True)
    
    # Step environment
    obs, reward, done, truncated, info = env.step(action)
    
    # Render
    env.render()
    
    # Slow it down so humans can see
    # (The AI thinks instantly, so without this the game finishes in 50ms)
    time.sleep(0.05)

    if done or truncated:
        print("Game Finished")
        obs, _ = env.reset()
        done = False