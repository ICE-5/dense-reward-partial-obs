import yaml
import argparse
import pathlib
from pathlib import Path

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
        "-o",
        "--out-dir",
        type=str,
        required=False,
        default=None,
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
    out_dir: Path,
    demo_names: list = None,
    split_only: bool = False,
):
    # available samplers
    sampler_map = {
        cls.__name__: cls for cls in (BlankSampler, BasicSampler, TemporalSampler)
    }
    sampler = sampler_map[config["sampler"]]

    if not split_only:
        spl = sampler(config=config, demo_path=demo_path, out_dir=out_dir)
        # spl = exec(config["sampler"])(config, demo_path, out_dir)
        # spl = sampler(config, demo_path, out_dir)

        prGreen(f"\nSUCCESS | init sampling with {config['sampler']}\n")
        spl.sample(demo_names)

    codes_path = out_dir / "codes.pkl"
    split_test_train(codes_path=codes_path, split_ratio=config["split_ratio"])
    prGreen(f"\nSUCCESS | train and test dataset generated at: {out_dir}\n")


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # if not specified, save output to data_dir
    out_dir = args.out_dir if args.out_dir else config["data_dir"]

    generate_dataset(
        config=config,
        demo_path=config["demo_path"],
        out_dir=Path(out_dir),
        demo_names=config["demo_names"],
        split_only=False,
    )
