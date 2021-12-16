# ==================================== |
# Name		    : 	Dalon Francis Lobo |
# Student ID	: 	202006328          |
# Email		    :  	x2020fyh@stfx.ca   |
# ==================================== |
import sys

from src.utils.file_helper import get_config

# globals
CONFIG_FILE = "config.toml"
# Load the configuration file
CONFIG = get_config(CONFIG_FILE)

if __name__ == "__main__":
    # Including imports inside the if conditions to speedup the execution
    # If all these includes are on top, they take a long time to load
    if CONFIG["runtime"]["mode"] == "FER-train":
        from src import FER_train

        FER_train.run_model(config=CONFIG)
    elif CONFIG["runtime"]["mode"] == "FER-predict":
        from src import FER_predict

        FER_predict.fer_predict(config=CONFIG)
    elif CONFIG["runtime"]["mode"] == "q2":
        from src import q2

        q2.explore_data(config=CONFIG)
    elif CONFIG["runtime"]["mode"] == "mt-q2":
        from src import q2

        q2.explore_mt_data(config=CONFIG)
    elif CONFIG["runtime"]["mode"] == "plot_model_history":
        from src.utils.plot_helper import plot_model_history

        plot_model_history(config=CONFIG)
    else:
        print("Error: Unknown runtime mode")
        sys.exit(-1)
