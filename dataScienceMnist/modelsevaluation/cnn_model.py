import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    Define the CNN model based in the architecture from the paper MNIST Handwritten
    Digit Recognition using Machine Learning by G, Elizabeth Rani et al.
    """

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)  # (28, 28, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # (14, 14, 32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)  # (14, 14, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # (7, 7, 64)
        self.flatten = nn.Flatten()  # (3136)
        self.fc1 = nn.Linear(7 * 7 * 64, 1024)  # (1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 10)  # (10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
