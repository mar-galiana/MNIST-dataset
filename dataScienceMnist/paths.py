import os

# Root directory of the project
ROOT_DIR = os.path.dirname(os.path.abspath(os.curdir))

# Directory to store trained models
MODELS_FOLDER = os.path.join(ROOT_DIR, "trained_models")

# Path to the trained CNN model file
CNN_MODEL_FILE = os.path.join(MODELS_FOLDER, "cnn_model.pickle")

# Path to the trained SVM model file
SVM_MODEL_FILE = os.path.join(MODELS_FOLDER, "svm_model.sav")

# Path to the trained KNN model file
KNN_MODEL_FILE = os.path.join(MODELS_FOLDER, "knn_model.sav")
