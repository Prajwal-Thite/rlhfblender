import importlib
import os
from typing import Optional, Union, List

import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecFrameStack,
    VecNormalize,
)

import gfootball.env as football_env
from rlhfblender.data_models.global_models import Environment
from rlhfblender.utils.utils import get_wrapper_class


def get_environment(
    env_name: str = "GFootball-11_vs_11_easy_stochastic-SMM-v0",
    n_envs: int = 1,
    environment_config: Optional[dict] = None,
    norm_env_path: Union[str, None] = None,
    additional_packages: list = (),
) -> VecEnv:
    """
    Get the Google Football environment by name.
    :param env_name: (str) Name of the environment
    :param n_envs: (int) Number of parallel environments
    :param additional_packages: (list) Additional packages to import
    :param environment_config: (dict) Environment configuration
    :param norm_env_path: (str) Path to the normalized environment
    :return: VecEnv
    """
    if environment_config is None:
        environment_config = {}
    for env_module in additional_packages:
        importlib.import_module(env_module)

    env_wrapper = get_wrapper_class(environment_config)
    env_id = 'GFootball-11_vs_11_easy_stochastic-SMM-v0'
    def create_gfootball_env():
        render_mode = environment_config.get('render', False)
        if render_mode:
            render_mode = "rgb_array"
        else:
            render_mode = "rgb_array"

        return football_env.create_environment(
            env_name=env_name,
            render="rgb_array",
            logdir=environment_config.get('logdir', '/tmp/football'),
            dump_frequency=environment_config.get('dump_frequency', 1),
            write_goal_dumps=environment_config.get('write_goal_dumps', False),
            write_full_episode_dumps=environment_config.get('write_full_episode_dumps', False),
            stacked=environment_config.get('stacked', True),
#            representation=environment_config.get('representation', 'extracted'),
            rewards=environment_config.get('rewards', 'scoring'),
            write_video=environment_config.get('write_video', False),
            number_of_left_players_agent_controls=environment_config.get('number_of_left_players_agent_controls', 1),
            number_of_right_players_agent_controls=environment_config.get('number_of_right_players_agent_controls', 0),
            channel_dimensions=environment_config.get('channel_dimensions', (96, 72)),
            other_config_options=environment_config.get('other_config_options', {})
        )

    vec_env_cls = DummyVecEnv
    env = make_vec_env(
        create_gfootball_env,
        n_envs=n_envs,
        wrapper_class=env_wrapper,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=environment_config.get("vec_env_kwargs", None),
        env_kwargs=environment_config.get("env_kwargs", None)   
    )

    if "vec_env_wrapper" in environment_config.keys():
        vec_env_wrapper = get_wrapper_class(environment_config, "vec_env_wrapper")
        env = vec_env_wrapper(env)

    # Load saved stats for normalizing input and rewards
    # And optionally stack frames
    if norm_env_path and environment_config.get("normalize", False):
        print("Loading running average")
        print(f"with params: {environment_config['normalize_kwargs']}")
        path_ = os.path.join(norm_env_path, "vecnormalize.pkl")
        if os.path.exists(path_):
            env = VecNormalize.load(path_, env)
            # Deactivate training and reward normalization
            env.training = False
            env.norm_reward = False
        else:
            raise ValueError(f"VecNormalize stats {path_} not found")

    n_stack = environment_config.get("frame_stack", 0)
    if n_stack > 0:
        print(f"Stacking {n_stack} frames")
        env = VecFrameStack(env, n_stack)

    return env


def initial_space_info(space: gym.spaces.Space) -> dict:
    """
    Get the initial space info for the environment, in particular the tag dict which is used for the
    the naming of the observation and action space in the user interface.
    :param space:
    :return: dict
    """
    shape = (space.n,) if isinstance(space, gym.spaces.Discrete) else space.shape

    tag_dict = {}
    if shape is not None:
        try:
            tag_dict = {f"{i}": i for i in range(shape[-1])}
        except:
            pass
    return {
        "label": f"{space.__class__.__name__}({shape!s})",
        "shape": shape,
        "dtype": str(space.dtype),
        "labels": tag_dict,
    }


def initial_registration(
    env_id: str = "11_vs_11_stochastic",
    entry_point: Optional[str] = "",
    additional_gym_packages: Optional[List[str]] = (),
    gym_env_kwargs: Optional[dict] = None,
) -> Environment:
    """
    Register the environment with the database.
    :param env_id: (str) The name of the environment
    :param entry_point: (Optional[str]) The entry point for the environment
    :param additional_gym_packages: (Optional[list]) Additional gym packages to import
    :param gym_env_kwargs: (Optional[dict]) Additional keyword arguments for the environment
    :return: Environment object
    """
    if len(additional_gym_packages) > 0:
        for env_module in additional_gym_packages:
            importlib.import_module(env_module)

    # Override the environment creation if it's Google Football
    env = football_env.create_environment(
        env_name=env_id,
        render=gym_env_kwargs.get('render', False),
        logdir=gym_env_kwargs.get('logdir', '/tmp/football'),
        dump_frequency=gym_env_kwargs.get('dump_frequency', 1),
        write_goal_dumps=gym_env_kwargs.get('write_goal_dumps', False),
        write_full_episode_dumps=gym_env_kwargs.get('write_full_episode_dumps', False),
        stacked=gym_env_kwargs.get('stacked', False),
        representation=gym_env_kwargs.get('representation', 'extracted'),
        rewards=gym_env_kwargs.get('rewards', 'scoring'),
        write_video=gym_env_kwargs.get('write_video', False),
        number_of_left_players_agent_controls=gym_env_kwargs.get('number_of_left_players_agent_controls', 1),
        number_of_right_players_agent_controls=gym_env_kwargs.get('number_of_right_players_agent_controls', 0),
        channel_dimensions=gym_env_kwargs.get('channel_dimensions', (96, 72)),
        other_config_options=gym_env_kwargs.get('other_config_options', {})
    )

    return Environment(
        env_name=env_id,
        registered=1,
        registration_id=env_id,
        observation_space_info=initial_space_info(env.observation_space),
        action_space_info=initial_space_info(env.action_space),
        has_state_loading=0,
        description="",
        tags=[],
        env_path="",
        additional_gym_packages=[] if len(additional_gym_packages) == 0 else additional_gym_packages,
    )
