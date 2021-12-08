# ==================================== |
# Name		    : 	Dalon Francis Lobo |
# Student ID	: 	202006328          |
# Email		    :  	x2020fyh@stfx.ca   |
# ==================================== |
import sys
from src.utils.file_helper import get_config
from src import q2

# globals
CONFIG_FILE = "config.toml"
# Load the configuration file
CONFIG = get_config(CONFIG_FILE)

if __name__ == "__main__":
    if CONFIG["runtime"]["mode"] == "train":
        pass
    elif CONFIG["runtime"]["mode"] == "test":
        pass
    elif CONFIG["runtime"]["mode"] == "q2":
        q2.explore_data(config=CONFIG)
    else:
        print("Error: Unknown runtime mode")
        sys.exit(-1)
