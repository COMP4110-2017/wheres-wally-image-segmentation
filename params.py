import env

# Training parameters
EPOCHS = env.EPOCHS
STEPS_PER_EPOCH = env.STEPS_PER_EPOCH
SPLIT = env.SPLIT
CHARACTER = env.CHARACTER

APPEND_MODEL = env.APPEND_MODEL
FROM_MODEL = env.FROM_MODEL
TO_MODEL = env.TO_MODEL

MODEL_NAME = CHARACTER + "_" + str(EPOCHS) + "_" + str(STEPS_PER_EPOCH) + "_" + str(SPLIT)

# Paths
IMAGE_PATH = "models/images/" + CHARACTER + "/raw_images/"
XML_PATH = "models/images/" + CHARACTER + "/bounding_boxes/"
TARGET_PATH = "models/images/" + CHARACTER + "/target_images/"
NEW_PATH = "models/images/" + CHARACTER + "/new_images/"
NUMPY_PATH = "models/images/" + CHARACTER + "/numpy/"
MODEL_PATH = "models/binaries/"
INPUT_PATH = "input/"
OUTPUT_PATH = "output/"
OUTPUT_PREFIX = FROM_MODEL + "_"

mu = 0.57417726724528806
std = 0.31263486676782137
