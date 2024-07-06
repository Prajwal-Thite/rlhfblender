import argparse
import asyncio
import os
import subprocess
import time
from typing import Dict, Optional

from databases import Database
import gymnasium as gym

import rlhfblender.data_collection.environment_handler as environment_handler
import rlhfblender.data_handling.database_handler as db_handler
from rlhfblender.data_models.global_models import Environment, Experiment, Project
from rlhfblender.utils.utils import StoreDict

database = Database(os.environ.get("RLHFBLENDER_DB_HOST", "sqlite:///rlhfblender.db"))

async def init_db():
    # Make sure all database tables exist
    await db_handler.create_table_from_model(database, Project)
    await db_handler.create_table_from_model(database, Experiment)
    await db_handler.create_table_from_model(database, Environment)

async def add_to_project(project: str = "RLHF-Blender", env: Optional[str] = None, exp: Optional[str] = None):
    """Add an environment or experiment to a project."""
    # Check if project exists
    if not await db_handler.check_if_exists(database, Project, key=project, key_column="project_name"):
        # Register new project
        await db_handler.add_entry(
            database,
            Project,
            Project(project_name=project, created_timestamp=int(time.time())).model_dump(),
        )
        existing_envs = []
        existing_exps = []
    else:
        # Get the project and existing envs and exps
        project_obj: Project = await db_handler.get_single_entry(database, Project, key=project, key_column="project_name")
        existing_envs = project_obj.project_environments
        existing_exps = project_obj.project_experiments

    # Now add env or exp to project
    if env is not None:
        await db_handler.update_entry(
            database,
            Project,
            key=project,
            key_column="project_name",
            data={"project_environments": [*existing_envs, env]},
        )
    if exp is not None:
        await db_handler.update_entry(
            database,
            Project,
            key=project,
            key_column="project_name",
            data={"project_experiments": [*existing_exps, exp]},
        )

async def register_env(
    id: str = "11_vs_11_stochastic",
    entry_point: Optional[str] = "",
    display_name: str = "",
    additional_gym_packages: Optional[list] = (),
    env_kwargs: Optional[Dict] = None,
    env_description: str = "",
    project: str = "RLHF-Blender",
):
    """Register an environment in the database and Gymnasium."""
    env_name = display_name if display_name != "" else id
    env_kwargs = env_kwargs if env_kwargs is not None else {}
    print(env_kwargs)
    print("id is  : ", id)
    print(entry_point)
    # Register the environment in Gymnasium
    gym.register(
        id="11_vs_11_stochastic",
        entry_point=entry_point,
        max_episode_steps=env_kwargs.get('max_episode_steps', None),
        kwargs=env_kwargs
    )

    env: Environment = environment_handler.initial_registration(
        env_id="11_vs_11_stochastic", entry_point=entry_point, additional_gym_packages=additional_gym_packages, gym_env_kwargs=env_kwargs
    )
    env.env_name = env_name
    env.description = env_description

    if not await db_handler.check_if_exists(database, Environment, key=id, key_column="registration_id"):
        await db_handler.add_entry(
            database,
            Environment,
            env.model_dump(),
        )
        await add_to_project(project=project, env=id)
        print(f"Registered environment {env_name} in project {project}")
    else:
        print(f"Environment with id {id} already exists. Skipping registration.")

async def register_experiment(
    exp_name: str,
    env_id: Optional[str] = "11_vs_11_stochastic",
    env_kwargs: Optional[Dict] = None,
    path: Optional[str] = "",
    framework: str = "StableBaselines3",
    exp_kwargs: Optional[Dict] = None,
    project: Optional[str] = "RLHF-Blender",
):
    """Register an experiment in the database."""
    env_kwargs = env_kwargs if env_kwargs is not None else {}
    exp_kwargs = exp_kwargs if exp_kwargs is not None else {}
    exp = Experiment(
        exp_name=exp_name, env_id=env_id, path=path, environment_config=env_kwargs, framework="StableBaselines3", **exp_kwargs
    )

    if not await db_handler.check_if_exists(database, Experiment, key=exp_name, key_column="exp_name"):
        await db_handler.add_entry(
            database,
            Experiment,
            exp.model_dump(),
        )
        await add_to_project(project=project, exp=exp_name)
        print(f"Registered experiment {exp_name} in project {project}")
    else:
        print(f"Experiment with name {exp_name} already exists. Skipping registration.")

async def get_action_dims(env_id: str) -> None:
    """Get the action dimensions for a given environment."""
    env: Environment = await db_handler.get_single_entry(database, Environment, key=env_id, key_column="registration_id")
    print(f"Action dimensions: {env.action_space_info.get('shape', 0)}")

async def register_action_labels(env_id: str, action_labels: list):
    """Register action labels for a given environment."""
    env: Environment = await db_handler.get_single_entry(database, Environment, key=env_id, key_column="registration_id")
    overwrite_action_space = env.action_space_info
    for key in overwrite_action_space["labels"]:
        overwrite_action_space["labels"][key] = action_labels[int(key)] if int(key) < len(action_labels) else key
    await db_handler.update_entry(
        database,
        Environment,
        key=env_id,
        key_column="registration_id",
        data={"action_space_info": overwrite_action_space},
    )

def generate_data(exp_name: str, num_episodes: int = 10, random: bool = True):
    """Generate data for the registered experiment."""
    cmd = [
        "python", "-m", "rlhfblender.generate_data",
        "--exp", exp_name,
        "--random" if random else "",
        #"-num-episodes", str(num_episodes)
    ]
    subprocess.run([arg for arg in cmd if arg])

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Register an environment or experiment in the database.")
    argparser.add_argument("--env", type=str, help="The environment id.", default="")
    argparser.add_argument("--exp", type=str, help="The experiment name.", default="")

    argparser.add_argument(
        "--project",
        type=str,
        help="(Optional) The project name. Defaults to RLHF-Blender.",
        default="RLHF-Blender",
    )

    argparser.add_argument(
        "--env-gym-entrypoint",
        type=str,
        help="(Optional) The gym entry point for the environment. Relevant for local custom environments",
        default="",
    )
    argparser.add_argument(
        "--env-display-name",
        type=str,
        help="(Optional) The display name for the environment. Relevant for local custom environments",
        default="",
    )
    argparser.add_argument(
        "--additional-gym-packages",
        type=str,
        nargs="+",
        help="(Optional) Additional gym packages to import. Relevant for local custom environments",
        default=[],
    )
    argparser.add_argument(
        "--env-kwargs",
        type=str,
        nargs="+",
        action=StoreDict,
        help='Environment Kwargs (e.g. description:"An optional env description")',
    )
    argparser.add_argument(
        "--env-description",
        type=str,
        help="(Optional) A description for the environment.",
        default="",
    )

    argparser.add_argument(
        "--exp-env",
        type=str,
        help="(Optional) A separate environment-id for the experiment. By default the environment-id is used.",
        default="",
    )
    argparser.add_argument(
        "--exp-path",
        type=str,
        help='(Optional) The path to the experiment. Defaults to "".',
        default="",
    )
    argparser.add_argument(
        "--exp-kwargs",
        type=str,
        nargs="+",
        action=StoreDict,
        help='Experiment Kwargs (e.g. description:"An optional exp description")',
    )

    argparser.add_argument(
        "--get-action-dims",
        action="store_true",
        help="Get the action dimensions for a given environment.",
        default=False,
    )
    argparser.add_argument(
        "--action-labels",
        type=str,
        nargs="+",
        help="(Optional) Action labels for the environment.",
        default=[],
    )

    args = argparser.parse_args()

    asyncio.run(init_db())

    if args.env != "":
        env_kwargs = args.env_kwargs if args.env_kwargs is not None else {}
        asyncio.run(
            register_env(
                args.env,
                entry_point=args.env_gym_entrypoint,
                display_name=args.env_display_name,
                additional_gym_packages=args.additional_gym_packages,
                env_kwargs=env_kwargs,
                env_description=args.env_description,
                project=args.project,
            )
        )        

    if args.exp != "":
        exp_kwargs = args.exp_kwargs if args.exp_kwargs is not None else {}
        exp_env_id = args.exp_env if args.exp_env != "" else args.env
        env_kwargs = args.env_kwargs if args.env_kwargs is not None else {}
        asyncio.run(
            register_experiment(
                args.exp,
                env_id=exp_env_id,
                env_kwargs=args.env_kwargs,
                path=args.exp_path,
                exp_kwargs=exp_kwargs,
                project=args.project,
            )
        )
        generate_data(exp_name=args.exp, random=True)

    if args.get_action_dims:
        asyncio.run(get_action_dims(args.env))

    if len(args.action_labels) > 0:
        asyncio.run(register_action_labels(args.env, args.action_labels))