import copy
import argparse
import pathlib
import csv
import pickle, yaml
import numpy as np

from dataloader.utils import Sample, FTWindow
from matplotlib import pyplot as plt

def generate_video(expert_rollout: list):
    l = len(expert_rollout)

    for step, sample in enumerate(expert_rollout):
        if step % 25 == 0 or step == l - 1:
            depthmap = sample.obs["depthmap"]
        
            # plt.imshow(depthmap, cmap="plasma")
            plt.imsave(f"debug/vis_expert_depthmap/depth_{step}.png", depthmap, cmap="plasma")

if __name__ == '__main__':
    with open("/Users/ice5/Documents/github/dense-reward-partial-obs/data/lap-joint/0mm/rd2/processed_expert_8_best.pkl", "rb") as f:
        expert_rollout = pickle.load(f)

    generate_video(expert_rollout)