import argparse
import yaml
from datetime import datetime

from dense_reward_partial_obs import DenseRewardPartialObs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path of configuration file",
    )
    parser.add_argument(
        "--model-params",
        type=str,
        required=False,
        help="if provided, will resume training based on given model params",
    )

    args = parser.parse_args()
    return args
    


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    model_id = datetime.now().strftime("%m%d%Y-%H%M%S")
    drpo = DenseRewardPartialObs(config, model_id=model_id, model_params=args.model_params)
    drpo.train()
