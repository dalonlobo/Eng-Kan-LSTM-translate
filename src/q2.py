"""
All the data exploration steps for q2
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing.image import load_img
from PIL import Image
import numpy as np
from sklearn.metrics import roc_auc_score
from src.utils.file_helper import create_en_kn_tiny
from src.utils.nlp_utils import get_clean_mtdata


def explore_data(config: dict) -> None:
    "Plot and print all the necessary data exploration steps"
    data_dir = Path().resolve() / config["dir"]["data"]
    train_path = data_dir / "train"
    test_path = data_dir / "test"
    class_names = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "sad", 6: "surprise"}
    train_count, test_count = {}, {}
    files_to_plot = {}
    for _, name in class_names.items():
        _path = train_path / name
        train_count[name] = len(list(_path.glob("*")))
        files_to_plot[name] = list(_path.glob("*"))[0]
        _path = test_path / name
        test_count[name] = len(list(_path.glob("*")))
    train_df = pd.DataFrame(train_count, index=["train"])
    test_df = pd.DataFrame(test_count, index=["test"])
    combined_df = pd.concat([train_df, test_df])
    print(combined_df.to_markdown())
    # Save the counts plot
    fig = combined_df.transpose().plot(kind="bar", fontsize=6, title="Number of images per emotion").get_figure()
    fig.savefig(Path().resolve() / config["dir"]["figures"] / "dataset_counts.png")
    # Save a plot with different emotions
    plt.figure(figsize=(10, 2))
    plt.tight_layout()
    for idx, (emotion, _path) in enumerate(files_to_plot.items()):
        img = load_img(_path)
        plt.subplot(1, 7, idx + 1)
        plt.imshow(img)
        plt.title(emotion)
        plt.axis("off")
    plt.savefig(Path().resolve() / config["dir"]["figures"] / "sample_images.png")

    # perform AUC analysis on images after resizing to 3x3
    # Choose 2 classes, angry = 0 and surprise = 1,
    # since these 2 of these clases are contrast different
    data1 = data_dir / "train" / "angry"
    data2 = data_dir / "train" / "surprise"
    resized_images = []
    bin_classes = []
    for bin_class, data in enumerate([data1, data2]):
        for filename in data.glob("*.jpg"):
            img = Image.open(filename)
            img = img.resize((3, 3), Image.ANTIALIAS)
            resized_images.append(np.asarray(img).ravel())
            bin_classes.append(bin_class)
        print(f"Set: {bin_class + 1} done")
    # convert to numpy
    resized_images = np.array(resized_images)
    bin_classes = np.array(bin_classes)
    COLS = ["Feature", "AUC"]
    aucs = pd.DataFrame(
        columns=COLS,
        data=np.zeros([9, len(COLS)]),
    )
    # Name each pixel of auc image
    features = [
        "top_left",
        "top_center",
        "top_right",
        "middle_left",
        "middle_center",
        "middle_right",
        "bottom_left",
        "bottom_center",
        "bottom_right",
    ]

    def _custom_sort(aucs: pd.Series) -> pd.Series:
        """custom sorting to sort AUC with weight for farther form 0.5

        Args:
            aucs (pd.Series): Column of aucs to sort.

        Returns:
            pd.Series: Sorted auc column.
        """

        def _sort(x: float) -> float:
            return (1 - x) if x < 0.5 else x

        ordered_aucs = list(aucs.unique())
        ordered_aucs.sort(key=_sort)
        map_dict = {item: idx for idx, item in enumerate(ordered_aucs)}
        return aucs.map(map_dict)

    for idx, feature in enumerate(features):
        aucs.iloc[idx] = (
            feature,
            roc_auc_score(bin_classes, resized_images[:, idx]),
        )
    aucs_sorted = aucs.sort_values(
        by="AUC",
        ascending=False,
        ignore_index=True,
        key=_custom_sort,
    )
    aucs_sorted["AUC"] = aucs_sorted["AUC"].round(3)
    # Print the results
    print(aucs_sorted.to_markdown())
    # create AUC image
    auc_image = aucs.AUC.to_numpy().reshape(3, 3)

    plt.figure()
    plt.imshow(auc_image, cmap="seismic")
    plt.title("Angry(0) vs Surprise(1)")
    plt.axis("off")
    plt.savefig(Path().resolve() / config["dir"]["figures"] / "AUC_image.png")

    print("Successfully saved auc image!")


def explore_mt_data(config: dict) -> None:
    """Explore the dataset for machine translation

    Args:
        config (dict): Configuration file

    Returns:
        None
    """
    data_dir = Path().resolve() / config["dir"]["data"]
    if config["mt"]["dataset"] == "en-kn-large":
        data_path = data_dir / "en-kn-large"
        train_en_size, train_kn_size = 0, 0
        print("Loading large data for analysis...")
        with open(data_path / "train.en", encoding="utf-8") as f1:
            for _ in f1:
                train_en_size += 1
        # Both files must have the same size
        train_kn_size = train_en_size
        print(f"Number of english sentences: {train_en_size}")
        print(f"Number of kannada sentences: {train_kn_size}")
        # Load the tiny en-kn data
        if (data_path / "en-kn-tiny.json").is_file():
            with open(data_path / "en-kn-tiny.json", encoding="utf-8") as f1:
                en_kn_tiny = json.load(f1)
            print("Successfully loaded: en-kn-tiny.json")
            print(f"Number of english sentences: {len(en_kn_tiny)}")
            print(f"Number of kannada sentences: {len(en_kn_tiny)}")
            # Get the cleaned data
            input_tensor, input_tokenizer, target_tensor, target_tokenizer = get_clean_mtdata(
                en_kn_tiny, max_sen_limit=config["mt"]["max_sen_token_size"]
            )
            print("Max english sentence length:", input_tensor.shape[1])
            print("Max kannada sentence length:", target_tensor.shape[1])
            print("English vocabulary size:", len(input_tokenizer.word_index))
            print("Kannada vocabulary size:", len(target_tokenizer.word_index))
        else:
            print("Creating tiny dataset")
            create_en_kn_tiny(config)
    else:
        print("Please select a known dataset for machine translation")
