import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class ModelEvaluator:
    """
    ModelEvaluator class to evaluate and visualize the performance of the models.

    This class handles the evaluation of both PyTorch and scikit-learn models,
    displaying accuracy and plotting confusion matrices.

    Attributes:
        data_manager (DataManager): An instance of DataManager to handle data operations.
    """

    def __init__(self, data_manager):
        """
        Initializes the ModelEvaluator with the given data manager.

        Args:
            data_manager (DataManager): An instance of DataManager to handle data operations.
        """
        self.data_manager = data_manager

    def evaluate_model(self, model, is_pytorch_model=False):
        """
        Evaluates the given model and displays its performance.

        This method predicts labels using the model, calculates accuracy, and
        plots the confusion matrix.

        Args:
            model (object): The machine learning model to be evaluated.
            is_pytorch_model (bool): Flag indicating if the model is a PyTorch model.
        """
        if is_pytorch_model:
            # Get predictions for a PyTorch model
            pred = self.get_predictions_cnn(model)
        else:
            # Get predictions for a scikit-learn model
            pred = self.get_predictions_sklearn_model(model)

        self.display_accuracy(pred)
        self.plot_table(pred)
        self.plot_conf_matrix(pred)

    def get_predictions_cnn(self, cnn_model):
        """
        This method processes the input data and gets predictions from the CNN model.

        Args:
            cnn_model (torch.nn.Module): The CNN model to get predictions from.

        Returns:
            numpy.ndarray: The predicted labels.
        """
        # Get sample data and convert to a PyTorch tensor
        x_data = self.data_manager.get_samples_data()
        x_tensor_data = torch.tensor(x_data, dtype=torch.float32).unsqueeze(1)

        # Get model predictions
        y_pred = cnn_model(x_tensor_data)
        predictions = torch.argmax(y_pred, 1).numpy()

        return predictions

    def get_predictions_sklearn_model(self, model):
        """
        This method processes the input data and gets predictions from the scikit-learn model.

        Args:
            model (sklearn.base.BaseEstimator): The scikit-learn model to get predictions from.

        Returns:
            numpy.ndarray: The predicted labels.
        """
        # Get sample data
        x_data = self.data_manager.get_samples_data()

        # Get model predictions
        predictions = model.predict(x_data)

        return predictions

    def display_accuracy(self, y_pred):
        """
        This method compares the predicted labels with the true labels and prints the accuracy.

        Args:
            y_pred (numpy.ndarray): The predicted labels.
        """
        y_true = self.data_manager.get_labels_data()

        # Calculate accuracy
        acc = (y_pred == y_true).sum() * 100 / len(y_true)

        print(f"Model's accuracy: {acc:.2f}%")

    def plot_conf_matrix(self, y_pred):
        """
        This method generates and displays a confusion matrix for the predicted and true labels.

        Args:
            y_pred (numpy.ndarray): The predicted labels.
        """
        print("Showing confusion matrix...")

        # Get true labels
        y_true = self.data_manager.get_labels_data()

        # Compute confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)

        # Display confusion matrix
        cmd = ConfusionMatrixDisplay(conf_matrix, display_labels=set(y_true))
        cmd.plot()
        plt.show()

    def plot_table(self, y_pred):
        """
        Plots a table showing the number of observations and accuracy rate per digit.

        Args:
            y_pred (numpy.ndarray): The predicted labels.
        """

        y_true = self.data_manager.get_labels_data()
        unique_labels = np.unique(y_true)

        # Print header for the table
        print("\nNumber of observations and accuracy rate per digit:")
        print("Digit\tObservations\tAccuracy")
        print("-"*35)

        # Iterate over each unique label (digit)
        for label in unique_labels:
            indices = np.where(y_true == label)
            observations = len(indices[0])

            correct_predictions = np.sum(y_pred[indices] == label)
            accuracy_rate = correct_predictions / observations * 100 if observations > 0 else 0
            print(f"{label}\t\t{observations}\t\t{accuracy_rate:.2f}%")

