import pathlib
import csv
import sys
import numpy as np
from matplotlib import pyplot as plt

def prGreen(skk):
    print("\033[92m {}\033[00m" .format(skk))


def prRed(skk):
    print("\033[91m {}\033[00m" .format(skk))

def prYellow(skk):
    print("\033[33m {}\033[00m" .format(skk))


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

if __name__ == '__main__':
    loss_file = sys.argv[1]
    model_id = pathlib.Path(loss_file).stem
    save_dir = pathlib.Path("media") / model_id / "vis_loss"
    save_dir.mkdir(parents=True, exist_ok=True)

    vis_loss(loss_file, save_dir)

    prGreen(f"\nSUCCESS | directory with plots: {save_dir}\n")


