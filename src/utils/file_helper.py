import sys
from pathlib import Path
from typing import Any, MutableMapping

import re
import toml
import json


def get_config(config_file: str) -> MutableMapping[str, Any]:
    """Read the configuration file in toml format

    Args:
        str (config_file): configuration filename

    Returns:
        dict: parsed toml file
    """
    # Load the configuration file
    config_path = Path().resolve() / config_file
    try:
        with open(config_path) as f:
            config = toml.load(f)
            print(f"Config file: {config_path} loaded!")
    except FileNotFoundError:
        print("Config file does not exist, create config.toml in working directory!")
        sys.exit(-1)
    return config


def create_en_kn_tiny(config: dict) -> None:
    """Creates tiny version of en-kn-large dataset with vocab from kn-tiny-vocab.json

    Args:
        config (dict): configuration
    """
    # load the vocab from kn-tiny-vocab.json
    data_dir = Path().resolve() / config["dir"]["data"]
    data_path = data_dir / "en-kn-large"
    with open(data_path / "kn-tiny-vocab.json", encoding="utf-8") as f:
        tiny_vocab = json.load(f)
    en_kn_tiny = []
    with open(data_path / "train.en", encoding="utf-8") as f1, open(data_path / "train.kn", encoding="utf-8") as f2:
        for idx, line in enumerate(f1):
            line = line.strip()
            kan_line = next(f2).strip()
            match = re.match("^[A-Za-z\s]*$", line)
            if match and (len(line.split()) == 5) and all(word in tiny_vocab for word in kan_line.split()):
                en_kn_tiny.append([line, kan_line])
    with open(data_path / "en-kn-tiny.json", "w", encoding="utf-8") as f:
        json.dump(en_kn_tiny, f)
    print("File en-kn-tiny.json written")
