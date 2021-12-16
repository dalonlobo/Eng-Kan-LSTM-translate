from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_model_history(config: dict) -> None:
    output_dir = Path().resolve() / config["dir"]["output"]
    fig_dir = Path().resolve() / config["dir"]["figures"]

    filename = input("Enter the logs filename: ")
    df = pd.read_csv(output_dir / filename, index_col=None, header=0)

    model_name = input("Enter model name: ")
    fig, ax = plt.subplots(1, 2)
    ax = ax.ravel()
    ax[0].plot(df.epoch, df.accuracy, label="train")
    ax[0].plot(df.epoch, df.val_accuracy, label="val")
    ax[0].set_title("Accuracy score")
    ax[0].legend()
    ax[1].plot(1, 2, 2)
    ax[1].plot(df.epoch, df.loss, label="train")
    ax[1].plot(df.epoch, df.val_loss, label="val")
    ax[1].set_title("Loss")
    ax[1].legend()
    fig.suptitle(f"{model_name} training")
    plt.tight_layout()
    model_name = model_name.replace(" ", "_")
    fig_path = fig_dir / f"{model_name}.png"
    plt.savefig(fig_path)
    print(f"Successfully saved the plot to: {fig_path}")


def plot_mtmodel_history(config: dict, history) -> None:
    "Plots the machine translation model training history"
    fig_dir = Path().resolve() / config["dir"]["figures"]
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title(f'Model: {config["mt"]["model_name"]}')
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    fig_path = fig_dir / f'{config["mt"]["model_name"]}.png'
    plt.savefig(fig_path)
    print(f"Successfully saved the plot to: {fig_path}")
