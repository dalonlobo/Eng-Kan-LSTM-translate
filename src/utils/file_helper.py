from pathlib import Path
import sys
import toml


def get_config(config_file: str) -> dict:
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
