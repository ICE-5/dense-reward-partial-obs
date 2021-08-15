import copy
import argparse
import pathlib
import csv
import pickle
import yaml
import numpy as np

from dataloader.utils import Sample, FTWindow


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path of configuration file, check configs/ for template",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="name of raw expert rollout .pkl & .csv file",
    )

    args = parser.parse_args()
    return args


def process_rollouts(
    config: dict,
    raw_expert_rollouts_path: str or None = None,
    selected_rollout_indices: list or None = None,
    num_output_rollouts: int or None = None,
    sort_by_length: bool = True,
    save: bool = False,
    save_name: str or None = None,
) -> list:

    if raw_expert_rollouts_path is None:
        raw_expert_rollouts_path = (
            config["data_dir"]
            / config["env_name"]
            / config["offset"]
            / "rd2"
            / "raw_expert.pkl"
        )
    raw_expert_rollouts_path = pathlib.Path(raw_expert_rollouts_path)
    if not raw_expert_rollouts_path.is_file():
        raise ValueError("Invalid raw expert rollouts path, file not exists")
    with open(raw_expert_rollouts_path, "rb") as reader:
        raw_expert_rollouts = pickle.load(reader)

    name = raw_expert_rollouts_path.stem
    reformatted_name = "_".join(name.split("_")[1:])

    output = []

    if selected_rollout_indices is not None:
        output_rollout_idxs = selected_rollout_indices
    else:
        output_rollout_idxs = list(range(len(raw_expert_rollouts)))

    for idx in output_rollout_idxs:
        rollout = raw_expert_rollouts[idx]
        reformatted_rollout = []

        ftw = FTWindow(initial_value=np.zeros([config["ft_window_size"], 6]),)
        for i, step in enumerate(rollout):
            # make sure the step contains info
            assert len(step) == 6

            if i == 0:
                ftw.update(step[0])
            ftw.update(step[2])

            obs = {}
            for sensor in config["sensor_used_in_sampling"]:
                if sensor == "ft":
                    obs["ft"] = copy.deepcopy(ftw.window)
                else:
                    obs[sensor] = step[5][sensor]

            sample = Sample(
                action=np.array(step[1]),
                obs=obs,
                pos=np.array(step[5]["pos"]),
                orn=np.array(step[5]["orn"]),
                dist_reward=step[3],
            )
            reformatted_rollout.append(sample)
        output.append(reformatted_rollout)

    # sort output by rollout length
    if sort_by_length:
        output.sort(key=len)

    # slice rollouts by given number
    if num_output_rollouts is not None:
        output = output[:num_output_rollouts]

    if save:
        if save_name is not None:
            save_name
        save_dir = config["data_dir"] / config["env_name"] / config["offset"] / "rd2"
        save_dir.mkdir(parents=True, exist_ok=True)
        pickle.dump(
            output,
            open(
                save_dir
                / f"processed_{reformatted_name}_{config['ft_window_size']}.pkl",
                "wb",
            ),
        )

    return output


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    config["data_dir"] = pathlib.Path(__file__).resolve().parent / "data"
    storage_dir = config["data_dir"] / config["env_name"] / config["offset"] / "rd2"

    # load successful rollout indices
    with open(storage_dir / f"{args.name}.csv") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            selected_rollout_indices = row
    selected_rollout_indices = [int(x) for x in selected_rollout_indices]

    output = process_rollouts(
        config,
        raw_expert_rollouts_path=storage_dir / f"{args.name}.pkl",
        selected_rollout_indices=selected_rollout_indices,
        num_output_rollouts=None,
        save=True,
    )

    reformatted_name = "_".join(args.name.split("_")[1:])

    # also save best exoert rollout
    save_dir = config["data_dir"] / config["env_name"] / config["offset"] / "rd2"
    pickle.dump(
        output[0],
        open(
            save_dir
            / f"processed_{reformatted_name}_{config['ft_window_size']}_best.pkl",
            "wb",
        ),
    )

    print(
        f"\n>>>>> finished processing, please check this directory for output: {save_dir}\n"
    )
