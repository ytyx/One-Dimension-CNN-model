# One Dimension CNN for Malware Classification

This repository contains code for classifying malware using 1D convolutional neural networks (CNNs). The process involves converting binary executable files into raw binary data, preprocessing the data, training a CNN model, and evaluating its performance.
<img width="729" alt="Screenshot 2024-03-13 at 11 01 45 AM" src="https://github.com/ytyx/One-Dimension-CNN-model/assets/87250788/c87263a2-62ba-47eb-b794-b7e54826d9e2">

- Data Representation: Converts binary executable files into raw binary data arrays.
- Dataset Preparation: Splits the dataset into training and testing sets, and creates DataLoader objects for PyTorch.
- Model Architecture: Defines the architecture of the 1D CNN model using PyTorch.
- Training: Train the CNN model using cross-entropy loss and the Adam optimizer.
- Evaluation: Evaluate the performance of the trained model on a test dataset.

### Binary Data Extraction
exe_to_binary.py script extracts byte values from binary executable files and saves them as raw binary data files. 
1. The script walks through the input directory and identifies all executable files.
2. It then creates a thread pool and assigns tasks to extract byte values from each file.
3. Extracted byte values are saved as raw binary data files in a separate directory.
4. Additional Note: The script can handle 7z compressed files if the necessary package is installed.

### Acknowledgments
Inspiration for the model architecture was derived from [Lin, W.-C.; Yeh, Y.-R. Efficient Malware Classification by Binary Sequences with One-Dimensional Convolutional Neural Networks. Mathematics 2022, 10, 608. https://doi.org/10.3390/math10040608].
