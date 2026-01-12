import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces
from random import randint

"""
The Engine (No Graphics)

Role: The Physics Engine.

Key Code: class TronEnv(ParallelEnv):

Goal: Inputs: Actions (0-4). Outputs: New Grid + Who Died.
"""

# --- MAP VALUE CONSTANTS ---
VAL_EMPTY  = 0
VAL_WALL   = 1
VAL_BOOST  = 2
VAL_MY_HEAD = 3  
VAL_ENEMY_HEAD = 4

# --- ACTION MAPPING CONSTANTS ---
ACT_UP    = 0
ACT_DOWN  = 1
ACT_LEFT  = 2
ACT_RIGHT = 3

TRAIL_OFF = 0
TRAIL_ON  = 1

BOOST_NO  = 0
BOOST_YES = 1

class TronEnv(ParallelEnv):
    def __init__(self):
        self.possible_agents = ["player_1", "player_2"]
        self.width = 100
        self.height = 100
        
        # 6 grid elements (channels): empty, wall, boost, my character, enemy character
        # each value will take value of 0 or 1, 0 or 255 for AI to process in CNN
        self.observation_spaces = {agent: spaces.Box(low=0, high=1, shape=(self.width, self.height, 5), dtype=np.uint8) for agent in self.agents}
        # 6 actions are: forward, backwards, left, right, boost, toggle trail.
        # 4 ways to steer, turn trail on or off, and boost or not boost (coast)
        self.action_spaces = {agent: spaces.MultiDiscrete([4, 2, 2]) for agent in self.agents}
        
    
    def reset(self):
        self.agents = self.possible_agents[:]
        self.grid = np.zeros((self.width, self.height), dtype=np.uint8)
        
        self.agent_positions = {}
        self.agent_dirs = {}      # Track current facing for Boosting
        self.trails_active = {}   # Toggle state
        self.boosts = {}     # Energy counter
        
        rows = self.grid.shape[0]
        cols = self.grid.shape[1]
        
        # create spawn points where there are no walls within 5 units
        self.grid[30, 50] = VAL_MY_HEAD
        self.agent_positions["player_1"] = (30, 50)
        self.agent_dirs["player_1"] = 2
        self.trails_active["player_1"] = True
        self.boosts["player_1"] = 0
        
        self.grid[70, 50] = VAL_ENEMY_HEAD
        self.agent_positions["player_2"] = (70, 50)
        self.agent_dirs["player_2"] = 4
        self.trails_active["player_1"] = True
        self.boosts["player_2"] = 0
        
        
        # create exterior walls
        for i in range(len(rows)):
            for j in range(len(cols)):
                if i == 0 or i == 99 or j == 0 or j == 99:
                    self.grid[i,j] = VAL_WALL
        
        # --- 2. GENERATE WALLS (Simplified) ---
        # Add Border Walls
        # self.grid[0, :] = VAL_WALL
        # self.grid[-1, :] = VAL_WALL
        # self.grid[:, 0] = VAL_WALL
        # self.grid[:, -1] = VAL_WALL
        
        # create 30 interior walls
        interior_walls = []
        
        wall_index = 0
        for k in range(30):
            if wall_index == 0:
                rand_x = randint(10, 20)
                rand_y = randint(10, 90)
            elif wall_index == 1:
                rand_x = randint(40, 60)
                rand_y = randint(10, 90)
            elif wall_index == 2:
                rand_x = randint(70, 90)
                rand_y = randint(10, 90)
                
            while((rand_x, rand_y) not in interior_walls):
                if wall_index == 0:
                    rand_x = randint(10, 20)
                    rand_y = randint(10, 90)
                elif wall_index == 1:
                    rand_x = randint(40, 60)
                    rand_y = randint(10, 90)
                elif wall_index == 2:
                    rand_x = randint(70, 90)
                    rand_y = randint(10, 90)
                    
            interior_walls.append((rand_x, rand_y))
            self.grid[rand_x, rand_y] =  VAL_WALL
            
            if wall_index != 2:
                wall_index += 1
            else:
                wall_index = 0
            
        # create 10 boost units
        boost_index = 0
        boost_units = []
        
        for k in range(10):
            if boost_index == 0:
                rand_x = randint(10, 20)
                rand_y = randint(10, 90)
            elif boost_index == 1:
                rand_x = randint(40, 60)
                rand_y = randint(10, 90)
            elif boost_index == 2:
                rand_x = randint(70, 90)
                rand_y = randint(10, 90)
                
            while((rand_x, rand_y) not in boost_units and (rand_x, rand_y) not in interior_walls):
                if boost_index == 0:
                    rand_x = randint(10, 20)
                    rand_y = randint(10, 90)
                elif boost_index == 1:
                    rand_x = randint(40, 60)
                    rand_y = randint(10, 90)
                elif boost_index == 2:
                    rand_x = randint(70, 90)
                    rand_y = randint(10, 90)
                
            interior_walls.append((rand_x, rand_y))
            self.grid[rand_x, rand_y] =  VAL_BOOST
            
            if boost_index != 2:
                boost_index += 1
            else:
                boost_index = 0
                
        return {a: self.observe(a) for a in self.agents}, {a: {} for a in self.agents}
        
        
        
    def step(self, actions):
        rewards = {}
        terminations = {}
        # (We assume everyone survives and gets 0 points initially)
        rewards = {a: 0 for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}
        
        died_this_step = []
        
        for agent in self.agents:
            
            for temp_agent in self.agents:
                if temp_agent != agent:
                    other_agent = temp_agent
                    break
            
            action = actions[agent]
            current_move_cmd = action[0]
            
            trail_cmd = action[1]
            self.trails_active[agent] = trail_cmd
            
            boost_cmd = action[2]
            if boost_cmd == BOOST_YES and self.boosts[agent] >= 1:
                # using one of the boosts
                self.boosts[agent] -= 1
            else:
                print("No more boosts left, can't use it")
            
            prev_move_cmd = self.agent_dirs[agent]
            final_move_cmd = current_move_cmd
            
            
            current_x, current_y = self.agent_positions[agent]
            
            # update grid based on movement (take into account the boost)
            dx, dy = 0, 0
            if boost_cmd == 0:
                move_distance = 1
                teleport = False
            elif boost_cmd == 1:
                move_distance = 3
                teleport = True

            if (prev_move_cmd == ACT_DOWN and current_move_cmd == ACT_UP) or (prev_move_cmd == ACT_UP and current_move_cmd == ACT_DOWN):
                final_move_cmd = prev_move_cmd
                rewards[agent] -= 0.1
                
            if(prev_move_cmd == ACT_LEFT and current_move_cmd == ACT_RIGHT) or (prev_move_cmd == ACT_RIGHT and current_move_cmd == ACT_LEFT):
                final_move_cmd = prev_move_cmd
                rewards[agent] -= 0.1
                
            if final_move_cmd == ACT_UP:
                dy += move_distance 
            elif final_move_cmd == ACT_DOWN:
                dy -= move_distance
            elif final_move_cmd == ACT_LEFT:
                dx -= move_distance
            elif final_move_cmd == ACT_RIGHT:
                dx += move_distance
            
            final_x, final_y = current_x + dx, current_y + dy
            
            # check if the final position is blocked by a wall or another player
            if self.grid[final_x, final_y] == VAL_WALL:
                terminations[agent] = True
                terminations[other_agent] = True
                died_this_step.append(agent)
                rewards[agent] -= 10
                
            elif self.grid[final_x, final_y] == VAL_ENEMY_HEAD:
                terminations[agent] = True
                rewards[agent] -= 5
                
            elif self.grid[final_x, final_y] == VAL_EMPTY:
                rewards[agent] += 0.05
                
            # update grid based on trail created behind the agent
            if teleport:
                self.grid[final_x, final_y] =  VAL_MY_HEAD
                
            elif not teleport and trail_cmd == TRAIL_ON:
                self.grid[current_x, current_y] = VAL_WALL
                
            # check if the player went over a boost, if so increase total boosts by 1
            # checking that the agent didn't die while going over the boost
            if agent in self.agents:
                if self.grid[final_x, final_y] == VAL_BOOST:
                    self.boosts[agent] += 1
                    
        if(len(died_this_step) == 1):
            # only 1 agent died so only 1 will get the positive reward of winning the game
            agent_dead = died_this_step[0]
            for temp_agent in self.possible_agents:
                if temp_agent != agent_dead:
                    agent_winner = temp_agent
                else:
                    self.agents.remove(temp_agent)
            # reward of 10 points for the agent winning
            rewards[agent_winner] += 10
        
        if(len(died_this_step) == 2):
            for dead_agent in died_this_step:
                self.agents.remove(dead_agent)
                
        # We only generate vision for agents who are still alive
        observations = {a: self.observe(a) for a in self.agents}
        return observations, rewards, terminations, truncations, infos
    
    
    def observe(self, agent):
        obs = np.zeros((self.width, self.height, 5), dtype=np.uint8)
        # creating an observation that is contains a 100 x 100 array for each type of grid component 
        # (wall (spawned in or generated by player), player 1 position, player 2 position,  energy boosts left on the map, energy dashboard (how many energy boosts picked up by a player))
        obs[:, :, 0] = (self.grid == self.VAL_WALL).astype(np.uint8) * 255
        if agent == "player_1":
            obs[:, :, 1] = (self.grid == self.VAL_MY_HEAD).astype(np.uint8) * 255
            obs[:, :, 2] = (self.grid == self.VAL_ENEMY_HEAD).astype(np.uint8) * 255
        elif agent == "player_2":
            obs[:, :, 1] = (self.grid == self.VAL_ENEMY_HEAD).astype(np.uint8) * 255
            obs[:, :, 2] = (self.grid == self.VAL_MY_HEAD).astype(np.uint8) * 255
        else:
            obs[:, :, 1] = (self.grid == self.VAL_MY_HEAD).astype(np.uint8) * 255
            obs[:, :, 2] = (self.grid == self.VAL_ENEMY_HEAD).astype(np.uint8) * 255
            
        obs[:, :, 3] = (self.grid == self.VAL_BOOST).astype(np.uint8) * 255
        
        # represents the number of energy boosts an agent has collected
        energy_boost_value = 255 * (self.boosts[agent]/10)
        obs[:, :, 4] = np.full((self.width, self.height), energy_boost_value, dtype=np.uint8)
        
        return obs
        
        
        
    def redner(self):
        pass
    
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    
    def action_space(self, agent):
        return self.action_spaces[agent]
    