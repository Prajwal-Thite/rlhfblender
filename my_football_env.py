import gymnasium as gym
from gfootball.env import create_environment
# from gymnasium.envs.registration import register
import numpy as np


# register(
#     id="MyFootballEnv-v0",
#     entry_point="my_football_env:FootballEnv",
# )


class FootballEnv(gym.Env):
    # metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, env_name="academy_pass_and_shoot_with_keeper", render_mode= 'rgb_array'):

        self.env = create_environment(env_name=env_name,rewards='scoring', 
                                      render_mode=render_mode, 
                                      render=True,                                      
                                      number_of_left_players_agent_controls=1,
                                      number_of_right_players_agent_controls=0,                                      
                                      )                                      
                                    #   other_config_options={'game_speed': 10000})
        
        # Set game speed through config after creation
        self.episode_steps = 0
        self.max_episode_steps = 300  # 30 seconds at 10 steps per second
        self.env.unwrapped._config['real_time'] = True
        self.env.unwrapped._config['video_quality_level'] = 2  # Higher quality rendering
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, seed=None, options=None):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self, mode="rgb_array"):
        return self.env.render(mode=mode)

    def close(self):
        return self.env.close()
