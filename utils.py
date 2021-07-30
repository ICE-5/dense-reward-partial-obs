import pathlib
import numpy as np
from matplotlib import pyplot as plt


def plot_curves(x: list or np.ndarray, ys: dict, save_dir: str, save_name: str) -> None:
    fig = plt.figure(figsize=(10, 5))
    for label, y in ys.items():
        plt.plot(x, y, label=label, linewidth=1)
    plt.legend()
    save_dir = pathlib.Path(save_dir)
    if "png" not in save_name:
        save_name += ".png"
    plt.savefig(save_dir / save_name)
    plt.close()

