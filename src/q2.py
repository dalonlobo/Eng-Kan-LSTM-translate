"""
All the data exploration steps for q2
"""
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img


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
