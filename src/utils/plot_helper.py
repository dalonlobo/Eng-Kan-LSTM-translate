from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf


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


def plot_feature_maps(config: dict, model: Any, data: Any) -> None:
    """Plots the feature maps for different layers

    Args:
        config (dict): configuration
        model (Any): model
        data (Any): test data
    """
    fig_dir = Path().resolve() / config["dir"]["figures"]

    layer_names = [layer.name for layer in model.layers]
    layer_outputs = [layer.output for layer in model.layers]
    feature_map_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    feature_maps = feature_map_model.predict(data.__getitem__(0)[0])

    images_per_row = 16
    _, axs = plt.subplots(10, figsize=(12, 10))
    fig_idx = 0
    for layer_name, layer_activation in zip(layer_names, feature_maps):  # Displays the feature maps
        if len(layer_activation.shape) == 4:
            n_features = layer_activation.shape[-1]  # Number of features in the feature map
            size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
            n_cols = n_features // images_per_row  # Tiles the activation channels in this matrix
            display_grid = np.zeros((size * n_cols, images_per_row * size))
            for col in range(n_cols):  # Tiles each filter into a big horizontal grid
                for row in range(images_per_row):
                    channel_image = layer_activation[0, :, :, col * images_per_row + row]
                    channel_image -= channel_image.mean()  # Post-processes the feature to make it visually palatable
                    channel_image /= channel_image.std()
                    channel_image *= 64
                    channel_image += 128
                    channel_image = np.clip(channel_image, 0, 255).astype("uint8")
                    display_grid[
                        col * size : (col + 1) * size, row * size : (row + 1) * size  # Displays the grid
                    ] = channel_image
            axs[fig_idx].title.set_text(layer_name)
            axs[fig_idx].imshow(display_grid[: display_grid.shape[0] // n_cols, :], aspect="auto", cmap="viridis")
            fig_idx += 1
    fig_path = fig_dir / "model-feature-map.png"
    plt.savefig(fig_path)
    print(f"Successfully saved the feature maps to: {fig_path}")


def plot_mtmodel_history(config: dict, history: Any) -> None:
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
