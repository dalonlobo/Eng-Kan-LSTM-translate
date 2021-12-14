# ==================================== |
# Name		    : 	Dalon Francis Lobo |
# Student ID	: 	202006328          |
# Email		    :  	x2020fyh@stfx.ca   |
# ==================================== |
import sys

from src import q2, FER_train, FER_predict
from src.utils.file_helper import get_config
from src.utils.plot_helper import plot_model_history

# globals
CONFIG_FILE = "config.toml"
# Load the configuration file
CONFIG = get_config(CONFIG_FILE)

if __name__ == "__main__":
    if CONFIG["runtime"]["mode"] == "train":
        FER_train.run_model(config=CONFIG)
    elif CONFIG["runtime"]["mode"] == "FER-predict":
        FER_predict.fer_predict(config=CONFIG)
    elif CONFIG["runtime"]["mode"] == "q2":
        q2.explore_data(config=CONFIG)
    elif CONFIG["runtime"]["mode"] == "plot_model_history":
        plot_model_history(config=CONFIG)
    else:
        print("Error: Unknown runtime mode")
        sys.exit(-1)
