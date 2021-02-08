import time

INPUT_SIZE_LIST = {"b0": 224, "b1": 240, "b2": 260, "b3": 300, "b4": 380, "b5": 456, "b6": 528, "b7": 600}

EPOCHS = 500
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

MODEL_BASE = "b4"
FIRST_LAYERS_TO_FREEZE = 147
INPUT_SIZE = INPUT_SIZE_LIST[MODEL_BASE]

TRAIN_IMAGE_FOLDER_PATH = "./data/train/images"
TRAIN_CSV_PATH = "./data/train/mcocr_train_df.csv"

OUTPUT_MODEL_PATH = f"./saved_model/{MODEL_BASE}_{str(time.time())}.h5"
OUTPUT_TRAINING_FIG_PATH = f"./plot_figure/{MODEL_BASE}_{str(time.time())}.jpg"

