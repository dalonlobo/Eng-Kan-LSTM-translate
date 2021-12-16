"""
Use the trained model to predict the Facial expression and map it to a emoji
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src import FER_train
from src.utils.plot_helper import plot_feature_maps


def fer_predict(config: dict) -> None:
    emotions = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    output_dir = Path().resolve() / config["dir"]["output"]
    model_dir = output_dir / config["dir"]["FER-model"]
    data_dir = Path().resolve() / config["dir"]["data"] / "test-dalon"
    emoji_dir = Path().resolve() / config["dir"]["data"] / "emojis"
    fig_dir = Path().resolve() / config["dir"]["figures"]

    # Get the FER model and load weights
    lobo_net = FER_train.get_model(config)
    lobo_net.summary()
    # Load the weights
    latest = tf.train.latest_checkpoint(model_dir)
    lobo_net.load_weights(latest)

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_set = test_datagen.flow_from_directory(
        data_dir,
        batch_size=1,
        target_size=tuple(config["img_info"]["size"]),
        shuffle=False,
        color_mode="grayscale",
        class_mode="categorical",
    )

    predictions = lobo_net.predict(test_set)
    _, axes = plt.subplots(nrows=2, ncols=6, figsize=(12, 4))
    ax = axes.ravel()
    for idx, filename in enumerate(test_set.filenames):
        pred_emo = np.argmax(predictions[idx])

        ax[idx].imshow(image.imread(data_dir / filename))
        ax[idx].set_title(emotions[pred_emo])
        ax[idx + 6].imshow(image.imread(emoji_dir / str(emotions[pred_emo] + ".png")))
        ax[idx].set_xticks([])
        ax[idx].set_yticks([])
        ax[idx + 6].set_xticks([])
        ax[idx + 6].set_yticks([])
    plt.tight_layout()
    plt.savefig(fig_dir / "fer-pred.png")

    # Plot feature maps for 1 prediction images
    plot_feature_maps(config, lobo_net, test_set)
