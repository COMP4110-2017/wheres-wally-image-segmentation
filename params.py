mu = 0.57417726724528806
std = 0.31263486676782137

EPOCHS = 1
STEPS_PER_EPOCH = 6
LOAD_MODEL = ""
SPLIT = 0
CHARACTER = "wanda"
IMAGE_PATH = "models/images/" + CHARACTER + "/raw_images/"
XML_PATH = "models/images/" + CHARACTER + "/bounding_boxes/"
TARGET_PATH = "models/images/" + CHARACTER + "/target_images/"
FULL_PREDICTIONS_PATH = "models/images/" + CHARACTER + "/full_predictions/"
NEW_PATH = "models/images/" + CHARACTER + "/new_images/"
NUMPY_PATH = "models/images/" + CHARACTER + "/numpy/"
MODEL_PATH = "models/binaries/"
SAVE_MODEL = "models/binaries/" + CHARACTER + "_" + str(EPOCHS) + "_" + str(STEPS_PER_EPOCH) + "_" + str(SPLIT) + ".h5"
