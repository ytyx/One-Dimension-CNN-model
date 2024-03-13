import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from exe_to_binary import *


# Step 1: Data Representation
def sep(folder_path):
    """
    Separates files into benign and malware categories
    from the specified folder path.
    """
    folder_path = 'data'
    benign_folder = os.path.join(folder_path, 'benign')
    malware_folder = os.path.join(folder_path, 'malware')
    benign_files = os.listdir(benign_folder)
    malware_files = os.listdir(malware_folder)
    return benign_files, malware_files


def conversion(binary_data):
    """
    This function resizes the binary data to a target length of 20000 bytes.
    If the length of the binary data is larger than or equal to 20000 bytes,
    it truncates the data to the first 20000 bytes to prevent information loss.
    If the length of the binary data is less than 20000 bytes, it pads the data
    with zeros to reach the target length.
    """
    resized_data = []
    # the target length should be larger to prevent info loss
    # kernal die if too large, maybe adjust on GPU
    if len(binary_data) >= 20000:
        resized_data.append(np.frombuffer(binary_data[:20000],
                                          dtype=np.uint8))
        # If data length is less than target length, pad with zeros
    else:
        padded_data = np.zeros(20000, dtype=np.uint8)
        padded_data[:len(binary_data)] = np.frombuffer(binary_data,
                                                       dtype=np.uint8)
        resized_data.append(padded_data)

    return resized_data


def create_dataset(benign_files, benign_folder, malware_files, malware_folder):
    """
    Create a dataset from benign and malware files.

    Args:
        benign_files (list): List of filenames in the 'benign' folder.
        benign_folder (str): Path to the 'benign' folder.
        malware_files (list): List of filenames in the 'malware' folder.
        malware_folder (str): Path to the 'malware' folder.
    """
    X = []  # Features
    y = []  # Labels
    # label assignment
    for file in benign_files:
        with open(os.path.join(benign_folder, file), 'rb') as f:
            binary_data = f.read()
            # Extract features from binary_data (you might need to use some
            # feature extraction technique)
            features = conversion(binary_data)
            X.append(features)
            y.append(0)  # 0 for benign

    for file in malware_files:
        with open(os.path.join(malware_folder, file), 'rb') as f:
            binary_data = f.read()
            # Extract features from binary_data and append to X
            features = conversion(binary_data)
            X.append(features)
            y.append(1)
    X = np.array(X)
    y = np.array(y)
    return X, y

# Step 2: Dataset Preparation


def create_tensor(X, y):
    """
    Create PyTorch tensors and DataLoader objects for training and testing.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).float()
    # Create DataLoader objects for training and testing
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader


# Step 3: Model Architecture
class ConvNet1D(nn.Module):
    def __init__(self):
        super(ConvNet1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=32, out_channels=16,
                               kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32,
                               kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64,
                               kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128,
                               kernel_size=3, stride=2, padding=1)
        self.pool4 = nn.MaxPool1d(kernel_size=2)

        self.conv5 = nn.Conv1d(in_channels=128, out_channels=32,
                               kernel_size=3, stride=2, padding=1)
        self.pool5 = nn.MaxPool1d(kernel_size=2)

        self.fc1 = nn.Linear(78, 32)
        self.fc2 = nn.Linear(32, 2)  # Output layer for binary classification

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.pool1(x)

        x = F.leaky_relu(self.conv2(x))
        x = self.pool2(x)

        x = F.leaky_relu(self.conv3(x))
        x = self.pool3(x)

        x = F.leaky_relu(self.conv4(x))
        x = self.pool4(x)
        x = F.leaky_relu(self.conv5(x))
        x = self.pool5(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = F.leaky_relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x


# Step 4: Training
def train1d(train_loader):
    '''
    Train a 1D convolutional neural network model.
    Found the best performed learning rate and number of epochs.
    '''
    model = ConvNet1D()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward + backward + optimize
            outputs = model(inputs.unsqueeze(1))
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
    print('Finished Training')

# Step 5: Evaluation


def model_eval(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.unsqueeze(1))  # Add a channel dimension
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted.unsqueeze(1) ==
                        labels.unsqueeze(1)).sum().item()

    accuracy = correct / total
    print(correct, total, f'Test Accuracy: {accuracy}')
