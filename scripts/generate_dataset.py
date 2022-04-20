import yaml
import argparse
import pathlib
from pathlib import Path

from drpo.samplers.sampler import Sampler
from drpo.samplers.basic_sampler import BasicSampler
from drpo.samplers.blank_sampler import BlankSampler
from drpo.samplers.temporal_sampler import TemporalSampler
from drpo.dataloader.utils import *
from drpo.utils import prGreen


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path of configuration file, check configs/ for template ",
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
        "--data-dir",
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
    demo_path: Path,
    data_dir: Path,
    demo_names: list = None,
    split_only: bool = False,
):
    
    if not split_only:
        spl = exec(sampler = config["sampler"])(config, demo_path, data_dir)
        # spl = sampler(config, demo_path, data_dir)
        spl.sample(demo_names)

    codes_path = data_dir / "codes.pkl"
    split_test_train(codes_path=codes_path, split_ratio=config["split_ratio"])
    prGreen(f"\nSUCCESS | train and test dataset generated at: {data_dir}\n")


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    generate_dataset(
        config=config,
        demo_path=Path(args.demo_path),
        data_dir=Path(args.data_dir),
        demo_names=[
            "demo_2",
        ],
        split_only=False,
    )
