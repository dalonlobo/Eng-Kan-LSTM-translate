# Machine translation training using various methods
import json
from pathlib import Path
from typing import Any

from keras.layers.embeddings import Embedding
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, GRU, LSTM
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from src.utils.plot_helper import plot_mtmodel_history
from src.utils.nlp_utils import get_clean_mtdata, logits_to_sentence


class MT_Models(object):
    def __init__(
        self, config: dict, model_name: str, ip_vocab_size: int, tr_vocab_size: int, input_shape: tuple
    ) -> None:
        self.config = config
        self.model_name = model_name
        self.ip_vocab_size = ip_vocab_size
        self.tr_vocab_size = tr_vocab_size
        self.input_shape = input_shape

    def simple_rnn_model(self) -> Any:
        "Returns the Simple RNN model created"
        model = Sequential()
        model.add(
            Embedding(
                self.ip_vocab_size,
                self.config["mt"]["embedding_size"],
                input_length=self.input_shape[1],
                input_shape=self.input_shape[1:],
            )
        )
        model.add(SimpleRNN(self.config["mt"]["num_rnn_units"], return_sequences=True))
        model.add(Dense(1024, activation="relu"))
        model.add(Dropout(self.config["mt"]["dropout_rate"]))
        model.add(Dense(self.tr_vocab_size, activation="softmax"))

        # Compile model
        model.compile(
            loss=sparse_categorical_crossentropy,
            optimizer=Adam(self.config["mt"]["learning_rate"]),
            metrics=["accuracy"],
        )
        return model

    def gru_model(self) -> Any:
        "Returns the GRU RNN model created"
        model = Sequential()
        model.add(
            Embedding(
                self.ip_vocab_size,
                self.config["mt"]["embedding_size"],
                input_length=self.input_shape[1],
                input_shape=self.input_shape[1:],
            )
        )
        model.add(GRU(self.config["mt"]["num_rnn_units"], return_sequences=True))
        model.add(Dense(1024, activation="relu"))
        model.add(Dropout(self.config["mt"]["dropout_rate"]))
        model.add(Dense(self.tr_vocab_size, activation="softmax"))

        # Compile model
        model.compile(
            loss=sparse_categorical_crossentropy,
            optimizer=Adam(self.config["mt"]["learning_rate"]),
            metrics=["accuracy"],
        )
        return model

    def gru_wo_embed_model(self) -> Any:
        "Returns the GRU RNN model created"
        model = Sequential()
        model.add(GRU(self.config["mt"]["num_rnn_units"], input_shape=self.input_shape[1:], return_sequences=True))
        model.add(Dense(1024, activation="relu"))
        model.add(Dropout(self.config["mt"]["dropout_rate"]))
        model.add(Dense(self.tr_vocab_size, activation="softmax"))

        # Compile model
        model.compile(
            loss=sparse_categorical_crossentropy,
            optimizer=Adam(self.config["mt"]["learning_rate"]),
            metrics=["accuracy"],
        )
        return model

    def lstm_model(self) -> Any:
        "Returns the GRU RNN model created"
        model = Sequential()
        model.add(
            Embedding(
                self.ip_vocab_size,
                self.config["mt"]["embedding_size"],
                input_length=self.input_shape[1],
                input_shape=self.input_shape[1:],
            )
        )
        model.add(LSTM(self.config["mt"]["num_rnn_units"], return_sequences=True))
        model.add(Dense(1024, activation="relu"))
        model.add(Dropout(self.config["mt"]["dropout_rate"]))
        model.add(Dense(self.tr_vocab_size, activation="softmax"))

        # Compile model
        model.compile(
            loss=sparse_categorical_crossentropy,
            optimizer=Adam(self.config["mt"]["learning_rate"]),
            metrics=["accuracy"],
        )
        return model

    def get_model(self) -> Any:
        "Helper to get the correct model"
        model_name = self.model_name.replace("-", "_")
        model = getattr(self, model_name)
        if hasattr(self, model_name) and callable(model):
            return model()
        else:
            print(f"Make sure {self.model_name} exists in the class")
            raise NotImplementedError(f"Model {self.model_name} is not implemented")


def train(config: dict) -> None:
    """Trains the selected model and displays the translated results

    Args:
        config (dict): configuration
    """
    data_dir = Path().resolve() / config["dir"]["data"]
    output_dir = Path().resolve() / config["dir"]["output"]
    data_path = data_dir / "en-kn-large"
    with open(data_path / "en-kn-tiny.json", encoding="utf-8") as f1:
        en_kn_tiny = json.load(f1)
    print("Successfully loaded: en-kn-tiny.json")
    # Get the cleaned data
    input_tensor, input_tokenizer, target_tensor, target_tokenizer = get_clean_mtdata(
        en_kn_tiny, max_sen_limit=config["mt"]["max_sen_token_size"]
    )
    # cross entropy needs labels in 3D
    target_tensor = target_tensor.reshape(*target_tensor.shape, 1)
    if config["mt"]["model_name"] == "gru-wo-embed-model":
        input_tensor = input_tensor.reshape((*input_tensor.shape, 1))
    model = MT_Models(
        config,
        config["mt"]["model_name"],
        len(input_tokenizer.word_index),
        len(target_tokenizer.word_index),
        input_tensor.shape,
    ).get_model()
    model.summary()
    history = model.fit(
        input_tensor[2:, :],
        target_tensor[2:, :, :],
        batch_size=config["mt"]["batch_size"],
        epochs=config["mt"]["epochs"],
        validation_split=config["mt"]["val_split"],
    )
    plot_mtmodel_history(config, history)
    # Show predictions
    results_path = output_dir / f'{config["mt"]["model_name"]}-results.txt'
    results = f"""
    Ground truth 1:\n\ten: {en_kn_tiny[0][0]}\n\tkn: {en_kn_tiny[0][1]}\n
    Prediction:\n\tkn: {logits_to_sentence(model.predict(input_tensor[:1])[0], target_tokenizer)}\n\n
    Ground truth 2:\n\ten: {en_kn_tiny[1][0]}\n\tkn: {en_kn_tiny[1][1]}\n
    Prediction:\n\tkn: {logits_to_sentence(model.predict(input_tensor[1:2])[0], target_tokenizer)}\n
    Model Params:\t
    {json.dumps(config["mt"], indent=4)}
    """
    with open(results_path, "w", encoding="utf8") as f:
        f.write(results)
    print(f"Successfully saved the results to: {results_path}")
