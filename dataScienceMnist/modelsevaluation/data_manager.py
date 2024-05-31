import numpy as np
from tensorflow.keras.datasets import mnist
from constants import BINARIZATION_THRESHOLD


class DataManager:
    """
    This class manages the MNIST dataset, allowing for random sampling, binarization,
    and formatting of data suitable for different types of models.

    Attributes:
        num_samples (int): Number of samples to load and manage.
        x_test (np.ndarray): Array containing the sample data.
        y_test (np.ndarray): Array containing the sample labels.
    """

    def __init__(self, num_samples: int):
        """
        Initializes the DataManager with the given number of samples.

        Args:
            num_samples (int): Number of samples to load and manage.
        """
        self.x_test = None
        self.y_test = None
        self.num_samples = num_samples
        self.__load_data()

    def get_samples_data(self):
        """
        Returns the sample data.

        Returns:
            np.ndarray: The sample data.
        """
        return self.x_test

    def get_labels_data(self):
        """
        Returns the sample labels.

        Returns:
            np.ndarray: The sample labels.
        """
        return self.y_test

    def __load_data(self):
        """
        Loads the MNIST dataset and randomly selects a subset of samples.
        """
        # Load the MNIST dataset
        __, (x_test, y_test) = mnist.load_data()

        # Select random observations from the dataset
        self.__get_random_observations(x_test, y_test)

    def __get_random_observations(self, x_test, y_test):
        """
        Selects random observations from the provided dataset.

        Args:
            x_test (np.ndarray): Array containing the test data.
            y_test (np.ndarray): Array containing the test labels.
        """
        # Select random indices for the specified number of samples
        indices = np.random.choice(len(y_test), self.num_samples, replace=False)

        # Assign the selected samples to the instance variables
        self.x_test = x_test[indices]
        self.y_test = y_test[indices]

    def process_data(self, is_pytorch_model=False):
        """
        This method binarizes the image data and reshapes it based on the model type.

        Args:
            is_pytorch_model (bool): Flag indicating if the model is a PyTorch model.
        """
        # Binarize the test images
        binary_images = self.__binarize_images()

        # Reshape the data if it is not for a PyTorch model
        if not is_pytorch_model:
            self.x_test = binary_images.reshape((binary_images.shape[0], -1))

    def __binarize_images(self):
        """
        Binarizes the images based on a predefined threshold.

        Returns:
            np.ndarray: The binarized images.
        """
        # Binarize images using the threshold
        binary_images = np.where(self.x_test > BINARIZATION_THRESHOLD, 1, 0)

        return binary_images
