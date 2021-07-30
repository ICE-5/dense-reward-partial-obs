import copy
import argparse
import pathlib
import csv
import pickle, yaml
import numpy as np

from dataloader.utils import Sample, FTWindow


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path of configuration file",
    )
    parser.add_argument(
        "--raw-expert-rollouts-pkl",
        type=str,
        default="data/lap-joint/expert_raw.pkl",
        help="path of raw rollouts .pkl file generated from RD2 project",
    )
    parser.add_argument(
        "--raw-expert-rollouts-csv",
        type=str,
        default="data/lap-joint/expert_raw.csv",
        help="path of .csv file specifying indices of successful rollouts in .pkl file",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="directory of the output",
    )
    parser.add_argument(
        "--save-name",
        type=str,
        default="expert.pkl",
        help="name of the output",
    )

    args = parser.parse_args()
    return args


def process_rollouts(
    config: dict,
    raw_expert_rollouts_pkl: str,
    raw_expert_rollouts_csv: str or None = None,
    save: bool = True,
    save_dir: str or None = None,
    save_name: str or None = None,
    sort_by_length: bool = True,
) -> None:

    successful_rollouts = []

    with open(raw_expert_rollouts_pkl, "rb") as reader:
        raw_rollouts = pickle.load(reader)

    if raw_expert_rollouts_csv is not None:
        with open(raw_expert_rollouts_csv, "r") as f:
            for row in csv.reader(f, delimiter=","):
                successful_idxs = [int(x) for x in row]
    else:
        successful_idxs = list(range(len(raw_rollouts)))

    for idx in successful_idxs:
        rollout = raw_rollouts[idx]
        reformatted_rollout = []

        ftw = FTWindow(
            initial_value=np.zeros([config["ft_window_size"], 6]),
        )
        for i, step in enumerate(rollout):
            # make sure the step contains info
            assert len(step) == 6

            if i == 0:
                ftw.update(step[0])
            ftw.update(step[2])

            # add ft sensor observation
            obs = {}
            obs["ft"] = copy.deepcopy(ftw.window)
            # add other sensor observation
            for sensor in config["sensor_used"]:
                if sensor != "ft":
                    obs[sensor] = step[5][sensor]


            sample = Sample(
                sample_name="expert",
                rollout_name=None,
                depth_name=None,
                action=np.array(step[1]),
                obs = obs,
                pos=np.array(step[5]["pos"]),
                orn=np.array(step[5]["orn"]),
                dist_reward=step[3],
            )
            reformatted_rollout.append(sample)
        successful_rollouts.append(reformatted_rollout)

    if sort_by_length:
        successful_rollouts.sort(key=len)

    if save:
        if save_dir is None:
            env_name = config["env_name"]
            data_dir = pathlib.Path(config["data_dir"])
            save_dir = data_dir / env_name
            # save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = pathlib.Path(save_dir)

        pickle.dump(successful_rollouts, open(save_dir / save_name, "wb"))

    return successful_rollouts


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    rollouts = process_rollouts(
        config,
        raw_expert_rollouts_pkl=args.raw_expert_rollouts_pkl,
        raw_expert_rollouts_csv=args.raw_expert_rollouts_csv,
        save_dir=args.save_dir,
        save_name=args.save_name,
        sort_by_length=True,
    )

    # sanity check: length of successful rollouts
    print([len(x) for x in rollouts])
    print([x[-1].dist_reward for x in rollouts])

