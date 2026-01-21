import pygame
import random
import sys # Added for safe exit
from tron_env import TronEnv 

# --- CONSTANTS ---
FPS = 60
MOVE_DELAY = 0.05  
ACT_UP, ACT_DOWN, ACT_LEFT, ACT_RIGHT = 0, 1, 2, 3

# 1. INIT PYGAME IMMEDIATELY
# We do this here to ensure the event system is ready before the loop starts
pygame.init()
pygame.display.set_caption("Tron RL Environment")

# 2. Setup Environment
env = TronEnv()
obs, infos = env.reset()

# 3. FORCE RENDERER CREATION
# We call render once to open the window immediately.
# This ensures that when the loop starts, the window exists to catch keys.
env.render() 

running = True
clock = pygame.time.Clock()
time_since_last_move = 0.0

# --- CONTROL STATE ---
next_move = ACT_LEFT 
next_boost = 0
next_trail = 1

print("Game Started. Use Arrow Keys to Move, Space to Boost.")

while running:
    # A. Tick Clock
    dt = clock.tick(FPS) / 1000.0 
    time_since_last_move += dt

    # B. EVENT HANDLING
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:    next_move = ACT_UP
            elif event.key == pygame.K_DOWN:  next_move = ACT_DOWN
            elif event.key == pygame.K_LEFT:  next_move = ACT_LEFT
            elif event.key == pygame.K_RIGHT: next_move = ACT_RIGHT
            elif event.key == pygame.K_SPACE: next_boost = 1
            elif event.key == pygame.K_t:     next_trail = 1 - next_trail
            elif event.key == pygame.K_ESCAPE: running = False

    # C. PHYSICS STEP
    if time_since_last_move >= MOVE_DELAY:
        
        p1_action = [next_move, next_trail, next_boost]
        p2_action = [random.randint(0, 3), 1, 0] 

        actions = { "player_1": p1_action, "player_2": p2_action }
        next_boost = 0 
        
        if env.agents:
            obs, rewards, terms, truncs, infos = env.step(actions)
            print(f"P1: move={next_move}, boost={p1_action[2]}, trail={next_trail} | Rewards: {rewards}")
            
            # Check Game Over
            if not env.agents:
                print("Game Over! All agents eliminated.")
                running = False
        
        time_since_last_move = 0.0  # FIXED: Reset to 0, not -= MOVE_DELAY

    # D. RENDER STEP
    env.render()

# Clean Exit
env.close()
pygame.quit()
sys.exit()