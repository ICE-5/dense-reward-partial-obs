import pathlib
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter1d
import time
import numpy as np
import pandas as pd
from itertools import cycle
import argparse
import csv
from numpy import genfromtxt
from numpy.random import choice


def prGreen(skk):
    print("\033[92m {}\033[00m".format(skk))


def prRed(skk):
    print("\033[91m {}\033[00m".format(skk))


def prYellow(skk):
    print("\033[33m {}\033[00m".format(skk))

def plot_smooth_curves(
    x: list or np.ndarray,
    ys: dict,
    save_dir: str,
    save_name: str,
    smoothing_window: int = 8,
    title: str = None,
    ys_max: dict = None,
    ys_min: dict = None,
    scale: float = 1.,
    xlabel: str = None,
    ylabel: str = None,
    font_size: int = 18,
):
    colors = ["#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#2ca02c", "#8c564b", "#e377c2", "#bcbd22", "#17becf"]

    linestyles = ["solid", "dashed", "dashdot", "dotted"]

    if not title:
        title = " "

    fig = plt.figure(figsize=(16, 8))
    color_index = 0

    ax = plt.subplot()  # Defines ax variable by creating an empty plot
    plt.ylim([0., 1.])

    # Set the tick labels font
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname("Arial")
        label.set_fontsize(font_size)

    for label, y in ys.items():

        y_smoothed = uniform_filter1d(y, size=smoothing_window) * scale

        # NOTE: caution
        # print(y_smoothed.shape)
        # y_smoothed[-10:] = 1.0

        # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.xaxis.get_offset_text().set_fontsize(font_size)


        # Running average for lines

        plt.plot(
            x,
            y_smoothed,
            label=label+" rewards",
            color=colors[color_index],
            ls=linestyles[0],
        )

        if ys_min:
            y_min = ys_min[label]
        
        

        y_min = ys_min[label]  if ys_min else 0
        y_max = ys_max[label]  if ys_max else np.max([y_smoothed, y])

        y_min_smoothed = uniform_filter1d(y_min, size=smoothing_window) * scale
        y_max_smoothed = uniform_filter1d(y_max, size=smoothing_window) * scale



        plt.fill_between(
            range(len(y)),
            y_min_smoothed,
            y_max_smoothed,
            # label=label,
            alpha=0.2,
            edgecolor=colors[color_index],
            facecolor=colors[color_index],
        )

        color_index += 1

    # plt.axhline(y=0.8, alpha=0.5, color='#000000', linestyle='dotted')

    axis_font = {"fontname": "Arial", "size": font_size}
    plt.legend(loc="lower right", prop={"size": font_size - 2})
    plt.xlabel(xlabel, **axis_font)
    plt.ylabel(ylabel, **axis_font)
    plt.title("%s" % title, **axis_font)

    # plt.show()
    # fig.savefig('%s.pdf' % title, dpi=fig.dpi, bbox_inches='tight')
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if "png" not in save_name:
        save_name += ".png"
    plt.savefig(save_dir / save_name, dpi=fig.dpi, bbox_inches="tight")
    plt.close()


def plot_curves(
    x: list or np.ndarray, ys: dict, title: str, save_dir: str, save_name: str
) -> None:
    fig = plt.figure(figsize=(7, 5))
    for label, y in ys.items():
        plt.plot(x, y, label=label, linewidth=1)
    plt.legend()
    plt.title(title)

    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if "png" not in save_name:
        save_name += ".png"
    plt.savefig(save_dir / save_name)
    plt.close()


def vis_loss(loss_file: str or pathlib.Path, save_dir: str) -> None:
    x = []
    train_loss, train_recon_loss, train_tmp_loss = [], [], []
    eval_loss, eval_recon_loss, eval_tmp_loss = [], [], []
    with open(loss_file, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for i, row in enumerate(reader):
            if i > 0:
                x.append(i)
                row = [float(x) for x in row]
                train_loss.append(row[0])
                train_recon_loss.append(row[1])
                train_tmp_loss.append(row[2])
                eval_loss.append(row[3])
                eval_recon_loss.append(row[4])
                eval_tmp_loss.append(row[5])

    plot_curves(
        x=x,
        ys={
            "combined loss": train_loss,
            "reconstruction loss": train_recon_loss,
            "temporal enforcement loss": train_tmp_loss,
        },
        title="Train",
        save_dir=save_dir,
        save_name=f"train_loss",
    )

    plot_curves(
        x=x,
        ys={
            "combined loss": eval_loss,
            "reconstruction loss": eval_recon_loss,
            "temporal enforcement loss": eval_tmp_loss,
        },
        title="Evaluation",
        save_dir=save_dir,
        save_name="eval_loss",
    )

    plot_curves(
        x=x,
        ys={
            "train": train_recon_loss,
            "eval": eval_recon_loss,
        },
        title="Reconstruction Loss",
        save_dir=save_dir,
        save_name=f"recon_loss",
    )

    plot_curves(
        x=x,
        ys={
            "train": train_tmp_loss,
            "eval": eval_tmp_loss,
        },
        title="Temporal enforcement Loss",
        save_dir=save_dir,
        save_name=f"tmp_loss",
    )

    plot_curves(
        x=x,
        ys={
            "train": train_loss,
            "eval": eval_loss,
        },
        title="Combined Loss",
        save_dir=save_dir,
        save_name=f"combined_loss",
    )


if __name__ == "__main__":
    loss_file = sys.argv[1]
    model_id = pathlib.Path(loss_file).stem
    save_dir = pathlib.Path("media") / model_id / "vis_loss"
    save_dir.mkdir(parents=True, exist_ok=True)

    vis_loss(loss_file, save_dir)

    prGreen(f"\nSUCCESS | directory with plots: {save_dir}\n")
