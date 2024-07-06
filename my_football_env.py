# In a file named my_football_env.py
import gymnasium as gym
from gfootball.env import create_environment

class FootballEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, env_name="11_vs_11_stochastic"):
        self.env = create_environment(env_name=env_name, render_mode='rgb_array')
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space             

    def reset(self, seed=None, options=None):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self, mode="rgb_array"):
        return self.env.render(mode=mode)

    def close(self):
        return self.env.close()
    

