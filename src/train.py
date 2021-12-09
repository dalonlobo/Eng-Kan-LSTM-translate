"CNN training and testing"
import os
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# use sequential keras model
def get_model(config: dict) -> tf.keras.Model:
    """construct the CNN model

    Args:
        config (dict): Configuration parameters

    Returns:
        tf.keras.Model: keras model
    """

    model = Sequential()
    model.add(
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            input_shape=tuple(config["img_info"]["size"] + [1]),
        )
    )
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.25))

    model.add(
        Conv2D(
            filters=128, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=regularizers.l2(0.01)
        )
    )
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu", kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(config["img_info"]["num_classes"], activation="softmax"))

    # Compliling the model
    model.compile(
        optimizer=Adam(learning_rate=config["cnn"]["learning_rate"], decay=1e-6),
        loss=config["cnn"]["loss"],
        metrics=config["cnn"]["metrics"],
    )
    return model


def get_callbacks(config: dict) -> list:
    output_dir = Path().resolve() / config["dir"]["output"]

    # Checkpoint the model for future use
    checkpoint = ModelCheckpoint(
        filepath=output_dir / config["cnn"]["chk_path"],
        save_best_only=True,
        verbose=1,
        mode="min",
        moniter="val_loss",
        save_weights_only=True,
    )

    # Since this is an experiment, stop training when validation loss does not improve
    # here patience is number of epochs without any improvements
    earlystop = EarlyStopping(monitor="val_loss", min_delta=0, patience=3, verbose=1, restore_best_weights=True)

    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=6, verbose=1, min_delta=0.0001)

    # Will not use tensorboard callback, since my system cannot handle the logging
    csv_logger = CSVLogger(output_dir / config["cnn"]["csv_log"])

    return [checkpoint, earlystop, reduce_lr, csv_logger]


def run_model(config: dict) -> None:
    data_dir = Path().resolve() / config["dir"]["data"]
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    # Load the dataset
    train_datagen = ImageDataGenerator(rescale=1.0 / 255, zoom_range=0.3, horizontal_flip=True)

    training_set = train_datagen.flow_from_directory(
        train_dir,
        batch_size=config["cnn"]["batch_size"],
        target_size=tuple(config["img_info"]["size"]),
        shuffle=True,
        color_mode="grayscale",
        class_mode="categorical",
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_set = test_datagen.flow_from_directory(
        test_dir,
        batch_size=config["cnn"]["batch_size"],
        target_size=tuple(config["img_info"]["size"]),
        shuffle=True,
        color_mode="grayscale",
        class_mode="categorical",
    )

    # get the model
    # naming the model after my second name
    lobo_net = get_model(config)
    lobo_net.summary()

    # steps has to be calculated
    steps_per_epoch = training_set.n // training_set.batch_size
    validation_steps = test_set.n // test_set.batch_size
    # Fit the model
    lobo_net.fit(
        x=training_set,
        validation_data=test_set,
        epochs=config["cnn"]["epoch"],
        callbacks=get_callbacks(config),
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
    )

    train_loss, train_accuracy = lobo_net.evaluate(training_set)
    test_loss, test_accuracy = lobo_net.evaluate(test_set)
    print(
        f"Train accuracy = {train_accuracy}\nTest accuracy = {test_accuracy}\n"
        f"Train loss = {train_loss}\nTest loss = {test_loss}"
    )
