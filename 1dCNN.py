import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from exe_to_binary import *

# Step 1: Data Representation


def sep(folder_path):
    folder_path = 'data/tiny_binary_files'
    benign_folder = os.path.join(folder_path, 'benign')
    malware_folder = os.path.join(folder_path, 'malware')
    benign_files = os.listdir(benign_folder)
    malware_files = os.listdir(malware_folder)
    return benign_files, malware_files

# byte conversion


def conversion(binary_data):

    eight_bit_vector = [byte for byte in binary_data]

    return np.array(eight_bit_vector)

# assign labels


def create_dataset(benign_files, benign_folder, malware_files, malware_folder):
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
    max_length = max(len(seq) for seq in X)  # Find the maximum length
    X_padded = [
        np.pad(
            seq,
            (0,
             max_length -
             len(seq)),
            mode='constant',
            constant_values=0) for seq in X]
    X_padded = np.array(X_padded)
    y = np.array(y)
    return X_padded, y

# Step 2: Dataset Preparation


def create_tensor(X_padded, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X_padded, y, test_size=0.2, random_state=42)
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
    def __init__(self, num_classes):
        super(ConvNet1D, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.conv4 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=1)
        self.pool4 = nn.MaxPool1d(kernel_size=2)

        self.conv5 = nn.Conv1d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=1)
        self.pool5 = nn.MaxPool1d(kernel_size=2)

        # Adjusted input size after max pooling
        self.fc1 = nn.Linear(128 * 24, 512)
        self.fc2 = nn.Linear(512, num_classes)

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
        x = x.view(-1, 128 * 24)  # Flatten the output
        x = F.leaky_relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)  # Softmax for classification
        return x

# Step 4: Training


def train1d(train_loader):
    model = ConvNet1D(2)
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
