mu = 0.57417726724528806
std = 0.31263486676782137

# Training parameters
EPOCHS = 2000
STEPS_PER_EPOCH = 6
SPLIT = 0.67
CHARACTER = "wanda"

# Paths
IMAGE_PATH = "models/images/" + CHARACTER + "/raw_images/"
XML_PATH = "models/images/" + CHARACTER + "/bounding_boxes/"
TARGET_PATH = "models/images/" + CHARACTER + "/target_images/"
NEW_PATH = "models/images/" + CHARACTER + "/new_images/"
NUMPY_PATH = "models/images/" + CHARACTER + "/numpy/"
MODEL_PATH = "models/binaries/"
SAVE_MODEL = "models/binaries/" + CHARACTER + "_" + str(EPOCHS) + "_" + str(STEPS_PER_EPOCH) + "_" + str(SPLIT) + ".h5"

# Predicting model (just the filename)
MODEL_NAME = "wanda_2000_6_0.67"
LOAD_MODEL = MODEL_NAME + ".h5"

INPUT_PATH = "input/"
OUTPUT_PATH = "output/"
OUTPUT_PREFIX = MODEL_NAME + "_"