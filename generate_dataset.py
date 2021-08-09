import yaml
import argparse
import pathlib
import pickle

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


def generate_dataset(config: dict, split_only: bool = False) -> None:
    # check env parameters in envs_launcher.py
    env = env_creator(None)

    # load and process expert demo
    expert_rollouts_path = (
        config["data_dir"]
        / config["env_name"]
        / config["offset"]
        / "rd2"
        / f"processed_expert_{config['ft_window_size']}.pkl"
    )
    if not expert_rollouts_path.is_file():
        raise Exception(
            "Missing processed expert rollouts, please run process_rollouts.py first"
        )
    with open(expert_rollouts_path, "rb") as f:
        expert_rollouts = pickle.load(f)
    num_expert_rollouts = (
        config["num_expert_rollouts"]
        if config["num_expert_rollouts"] < len(expert_rollouts)
        else len(expert_rollouts)
    )
    expert_rollouts = expert_rollouts[:num_expert_rollouts]

    # specify output dir by dataset name
    output_dir = (
        config["data_dir"]
        / config["env_name"]
        / config["offset"]
        / config["dataset_name"]
    )
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n>>>>> using sampler: {config['sampler']}\n")
    sampler = eval(config["sampler"])(
        config,
        env=env,
        expert_rollouts=expert_rollouts,
        output_dir=output_dir,
    )

    # sample and split
    if not split_only:
        sampler.sample()

    sampler.split_train_test()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    config["data_dir"] = pathlib.Path(__file__).resolve().parent / "data"

    generate_dataset(config, args.split_only)
