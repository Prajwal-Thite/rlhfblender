from gfootball.env import create_environment
import gym
import numpy as np

class FootballEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, env_name="academy_pass_and_shoot_with_keeper", render_mode='rgb_array'):
        self.env = create_environment(
            env_name=env_name,
            rewards='scoring',
            representation='raw',
            render=True,
            write_video=True,
            write_full_episode_dumps=True,
            logdir='football/logs',
            number_of_left_players_agent_controls=1,
            number_of_right_players_agent_controls=0                                    
        )                                      
        
        # Initialize spaces after environment creation
        self._observation_space = self.env.observation_space
        self._action_space = self.env.action_space
        
        self.episode_steps = 0
        self.max_episode_steps = 300

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def reset(self):
        obs = self.env.reset()
        self.episode_steps = 0
        return obs

    def step(self, action):
        self.episode_steps += 1
        obs, reward, done, info = self.env.step(action)
        if self.episode_steps >= self.max_episode_steps:
            done = True
        return obs, reward, done, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

if __name__ == "__main__":
    env = FootballEnv()
    obs = env.reset()
    done = False
    
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
    
    env.close()
