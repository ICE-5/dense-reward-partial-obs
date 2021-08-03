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
        "--config", type=str, required=True, help="path of configuration file",
    )

    args = parser.parse_args()
    return args


def process_rollouts(
    config: dict,
    raw_expert_rollouts_path: str or None = None,
    num_output_rollouts: int or None = None,
    selected_rollout_indices: list or None = None,
    save: bool = False,
) -> None:

    if raw_expert_rollouts_path is None:
        raw_expert_rollouts_path = (
            config["data_dir"]
            / config["env_name"]
            / config["offset"]
            / "rd2"
            / "expert_raw.pkl"
        )
    raw_expert_rollouts_path = pathlib.Path(raw_expert_rollouts_path)
    if not raw_expert_rollouts_path.is_file():
        raise ValueError("Invalid raw expert rollouts path, file not exists")
    with open(raw_expert_rollouts_path, "rb") as reader:
        raw_expert_rollouts = pickle.load(reader)

    output = []

    if selected_rollout_indices is not None:
        output_rollout_idxs = selected_rollout_indices
    else:
        raw_expert_rollouts.sort(key=len)
        if num_output_rollouts is None:
            num_output_rollouts = len(raw_expert_rollouts)
        output_rollout_idxs = list(range(num_output_rollouts))

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
            for sensor in config["sensor_used"]:
                if sensor == "ft":
                    obs["ft"] = copy.deepcopy(ftw.window)
                else:
                    obs[sensor] = step[5][sensor]

            sample = Sample(
                sample_name="expert",
                rollout_name=None,
                depth_name=None,
                action=np.array(step[1]),
                obs=obs,
                pos=np.array(step[5]["pos"]),
                orn=np.array(step[5]["orn"]),
                dist_reward=step[3],
            )
            reformatted_rollout.append(sample)
        output.append(reformatted_rollout)

    if save:
        save_dir = (
            config["data_dir"]
            / config["env_name"]
            / config["offset"]
            / config["dataset_name"]
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        pickle.dump(output, open(save_dir / "expert.pkl", "wb"))
    
    if num_output_rollouts == 1:
        output = output[0]

    return output


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    config["data_dir"] = pathlib.Path(__file__).resolve().parent / "data"


    process_rollouts(
        config,
        num_output_rollouts=config["num_expert_rollouts"],
        save=True,
    )

