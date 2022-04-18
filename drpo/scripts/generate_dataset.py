import yaml
import argparse
import pathlib
import pickle

# from samplers.naive_backward_sampler import NaiveBackwardSampler
# from samplers.temporal_variant_backward_sampler import TemporalVariantBackwardSampler
# from samplers.temporal_variant_forward_sampler import TemporalVariantForwardSampler
# from envs.envs_launcher import env_creator
# from utils import prGreen, prYellow

from drpo.samplers.sampler import Sampler
from drpo.samplers.basic_sampler import BasicSampler
from drpo.dataloader.utils import *
from drpo.utils import prGreen, prYellow


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path of configuration file, check drpo/configs/ for template ",
    )
    parser.add_argument(
        "-d",
        "--demo-path",
        type=str,
        required=True,
        help="path of demo hdf5 file",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=str,
        required=True,
        help="directory of output",
    )
    parser.add_argument(
        "--split-only",
        type=bool,
        default=False,
        help="if set to true, only split train/test set without regenerating samples",
    )

    args = parser.parse_args()
    return args


def generate_dataset(
    config: dict,
    sampler: Sampler,
    demo_path: pathlib.Path,
    out_dir: pathlib.Path,
    split_only: bool = False,
):
    if not split_only:
        spl = sampler(config, demo_path, out_dir)
        demo_names = ["demo_2", ]
        spl.sample(demo_names)
    
    codes_path = out_dir / "codes.pkl"
    split_test_train(codes_path=codes_path, split_ratio=config["split_ratio"])
    prGreen(f"\nSUCCESS | train and test dataset: {out_dir}\n")



# def generate_dataset(
#     config: dict, expert_rollouts_name: str or None = None, split_only: bool = False
# ) -> None:
#     # check env parameters in envs_launcher.py
#     env = env_creator(None)

#     if expert_rollouts_name is None:
#         raise ValueError("please specify the name of processed expert rollout file")
#     else:
#         expert_rollouts_name = pathlib.Path(expert_rollouts_name).stem
#         if "processed" in expert_rollouts_name:
#             expert_rollouts_name = "_".join(expert_rollouts_name.split("_")[1:])

#     # load and process expert demo
#     expert_rollouts_path = (
#         config["data_dir"]
#         / config["env_name"]
#         / config["offset"]
#         / "rd2"
#         / f"processed_{expert_rollouts_name}_{config['ft_window_size']}.pkl"
#     )
#     if not expert_rollouts_path.is_file():
#         raise FileNotFoundError(
#             "Missing processed expert rollouts, try run process_rollouts.py first"
#         )
#     with open(expert_rollouts_path, "rb") as f:
#         expert_rollouts = pickle.load(f)

#     num_expert_rollouts = (
#         config["num_expert_rollouts"]
#         if config["num_expert_rollouts"] < len(expert_rollouts)
#         else len(expert_rollouts)
#     )
#     expert_rollouts = expert_rollouts[:num_expert_rollouts]

#     # specify output dir by dataset name
#     output_dir = (
#         config["data_dir"]
#         / config["env_name"]
#         / config["offset"]
#         / config["dataset_name"]
#     )
#     pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

#     prYellow(f"\nINITIATED | using sampler: {config['sampler']}\n")
#     sampler = eval(config["sampler"])(
#         config, env=env, expert_rollouts=expert_rollouts, output_dir=output_dir,
#     )

#     # sample and split
#     if not split_only:
#         sampler.sample()

#     sampler.split_train_test()

#     prGreen(f"\nSUCCESS | processed expert rollouts in: {output_dir}\n")


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    config["data_dir"] = pathlib.Path(__file__).resolve().parent / "data"

    generate_dataset(config, args.expert_rollouts_name, args.split_only)
