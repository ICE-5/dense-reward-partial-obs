import argparse
from datetime import datetime
import yaml
import pathlib

from drpo.envs.envs_launcher import env_creator
from robosuite.wrappers.gym_wrapper import GymWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path of configuration file, check configs/template.yaml",
    )
    # parser.add_argument(
    #     "--model-params-path",
    #     type=str,
    #     required=False,
    #     default=None,
    #     help="if provided, will resume training based on given model params",
    # )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    demo_path = pathlib.Path(config["demo_path"])
    benchmark_dir = pathlib.Path("benchmark")

    env, env_name = env_creator(demo_path)
    env = GymWrapper(env)

    model_id = datetime.now().strftime("%m%d%Y-%H%M%S")

    model = SAC(
        policy="MlpPolicy",
        env=env,
        optimize_memory_usage=True,
        buffer_size=300,
        verbose=1,
        tensorboard_log=benchmark_dir,
    )
    model.learn(total_timesteps=10000, log_interval=1, tb_log_name=model_id)
    model.save(benchmark_dir / "test")

    # model.load()

    # obs = env.reset()
    # print(type(obs), obs.shape)

    # obs_space = env.observation_space
    # print(obs_space.keys())
    # check_env(env)
    pass
