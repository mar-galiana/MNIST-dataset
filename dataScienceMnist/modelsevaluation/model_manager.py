import pickle
from enum import Enum
from cnn_model import CNN
from data_manager import DataManager
from model_evaluator import ModelEvaluator
from paths import CNN_MODEL_FILE, KNN_MODEL_FILE, SVM_MODEL_FILE


class Classifiers(Enum):
    """
    Enum class for classifiers.

    This enumeration defines the types of classifiers supported.
    """
    CNN = 1
    SVM = 2
    KNN = 3


class ModelManager:
    """
    ModelManager class to manage the loading and evaluation of models.

    This class handles the logic for loading different types of models
    (CNN, SVM, KNN) and evaluating their performance.

    Attributes:
        clf_type (Classifiers): The type of classifier to be managed.
        data_manager (DataManager): An instance of DataManager to handle data processing.
        model_evaluator (ModelEvaluator): An instance of ModelEvaluator to evaluate model performance.
        model (object): The trained model to evaluate.
    """

    def __init__(self, clf_type: Classifiers, data_manager: DataManager):
        """
        Initializes the ModelManager with the given classifier type and data manager.

        Args:
            clf_type (Classifiers): The type of classifier to be managed.
            data_manager (DataManager): An instance of DataManager to handle data processing.
        """
        self.model = None
        self.clf_type = clf_type
        self.data_manager = data_manager
        self.model_evaluator = ModelEvaluator(self.data_manager)

    def load_model(self):
        """
        Loads the model based on the classifier type.

        This method loads a CNN model from a file using PyTorch, or
        loads SVM/KNN scikit-learn models.
        """
        if self.clf_type == Classifiers.CNN:
            self.model = CNN()

            with open(CNN_MODEL_FILE, "rb") as fp:
                self.model.load_state_dict(pickle.load(fp))

        else:
            filename = KNN_MODEL_FILE if self.clf_type == Classifiers.KNN else SVM_MODEL_FILE
            self.model = pickle.load(open(filename, 'rb'))

    def show_model_performance(self):
        """
        Shows the performance of the loaded model.

        This method processes the data, evaluates the model, and displays its performance.
        """

        is_pytorch_model = self.clf_type == Classifiers.CNN
        self.data_manager.process_data(is_pytorch_model=is_pytorch_model)
        self.model_evaluator.evaluate_model(self.model, is_pytorch_model=is_pytorch_model)
