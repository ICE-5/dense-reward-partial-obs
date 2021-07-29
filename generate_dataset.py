import yaml
import argparse
import pathlib

from samplers.naive_backward_sampler import NaiveBackwardSampler
from samplers.temporal_variant_backward_sampler import TemporalVariantBackwardSampler
from envs.envs_launcher import env_creator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/debug.yaml",
        help="path of configuration file",
    )
    parser.add_argument(
        "--split-only",
        type=bool,
        default=False,
        help="if set to true, only split train/test set without regenerating samples",
    )

    args = parser.parse_args()
    return args


def generate_dataset(config, split_only: bool = False) -> None:
    env = env_creator(None)
    env_name = config["env_name"]
    data_dir = pathlib.Path(config["data_dir"])

    expert_demo_path = data_dir / env_name / "expert.pkl"
    output_dir = data_dir / env_name / "samples"

    # sample and split
    if split_only:
        sampler = eval(config["sampler"])(
            config,
            env=env,
            expert_demo_path=expert_demo_path,
            output_dir=output_dir,
            empty_output_dir=False,
        )
    else:
        sampler = eval(config["sampler"])(
            config,
            env=env,
            expert_demo_path=expert_demo_path,
            output_dir=output_dir,
            empty_output_dir=True,
        )
        sampler.sample()

    sampler.split_train_test()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    generate_dataset(config, args.split_only)
