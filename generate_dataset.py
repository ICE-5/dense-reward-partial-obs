import yaml
import argparse
import pathlib

from samplers.naive_backward_sampler import NaiveBackwardSampler
from samplers.temporal_variant_backward_sampler import TemporalVariantBackwardSampler
from envs.envs_launcher import env_creator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="path of configuration file",
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

    dataset_dir = (
        config["data_dir"]
        / config["env_name"]
        / config["offset"]
        / config["dataset_name"]
    )

    expert_rollouts_path = dataset_dir / "expert.pkl"
    if not expert_rollouts_path.is_file():
        raise Exception(
            "Missing processed expert rollouts, please run process_rollouts.py first"
        )

    print(f"\nUsing sampler: {config['sampler']}\n")

    # sample and split
    if split_only:
        sampler = eval(config["sampler"])(
            config,
            env=env,
            expert_rollouts_path=expert_rollouts_path,
            empty_output_dir=False,
        )
    else:
        sampler = eval(config["sampler"])(
            config,
            env=env,
            expert_rollouts_path=expert_rollouts_path,
            empty_output_dir=True,
        )
        sampler.sample()

    sampler.split_train_test()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    config["data_dir"] = pathlib.Path(__file__).resolve().parent / "data"

    generate_dataset(config, args.split_only)
