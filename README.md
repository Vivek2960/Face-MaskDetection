# Face Mask Detection

## Overview

This project is a Face Mask Detection system that uses deep learning techniques to identify whether a person is wearing a face mask or not. It is built using TensorFlow and Keras, with a Convolutional Neural Network (CNN) model trained on a dataset containing images of people with and without masks.

## Features

- Detects if a person is wearing a face mask or not.
- High accuracy with real-time detection capability.
- Easy to deploy on various platforms, including local machines and cloud services.

## Project Structure

- `model.py`: Script to train and evaluate the mask detection model.
- `convert_model.py`: Script to convert and save the model in different formats.
- `mask_detector.model`: The pre-trained model.
- `datasets/`: Directory containing training and validation datasets.
- `requirements.txt`: Contains the list of dependencies needed for the project.

## Requirements

### Hardware Requirements

- **GPU**: NVIDIA GPU (Optional but recommended for faster training)
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: At least 5GB of free disk space

### Software Requirements

- **Python 3.7+**
- **TensorFlow 2.17.0**
- **Keras 3.4.1**
- **OpenCV**
- **imutils**
- **scikit-learn**

### Python Packages

## Libraries and Dependencies

### Core Libraries

- **TensorFlow/Keras:**
  - **tensorflow**: The foundation for building and training the neural network model.
  - **keras**: A high-level API built on top of TensorFlow, making it easier to work with neural networks.
  - **Used for:**
    - `load_model`: Loads your trained mask detection model.
    - **Model building blocks**: `Input`, `AveragePooling2D`, `Flatten`, `Dense`, `Dropout`.
    - **Data augmentation**: `ImageDataGenerator`.
    - **Optimizers**: `Adam`.

- **OpenCV (cv2):**
  - A powerful computer vision library used for real-time image and video processing.
  - **Used for:**
    - **Video capture**: `VideoCapture`, `VideoStream`.
    - **Image manipulation**: `imread`, `resize`, `cvtColor`.
    - **Face detection**: `dnn` module (Deep Neural Network).
    - **Drawing on images**: `rectangle`, `putText`.

### Helper Libraries

- **imutils:**
  - A collection of convenience functions to make basic image processing operations easier with OpenCV.
  - **Used for:**
    - **Image resizing**: `resize` to resize images/frames while maintaining aspect ratio.
    - **Video handling**: `VideoStream` for simplified video stream handling.

- **NumPy (np):**
  - Fundamental library for numerical computing in Python.
  - **Used for:**
    - **Array operations**: Creating and manipulating arrays (e.g., storing images, processing model outputs).

- **Matplotlib (plt):**
  - A plotting library for creating static, interactive, and animated visualizations.
  - **Used for:**
    - **Plotting training curves**: Visualize loss and accuracy during model training.
    - **Model evaluation**: Potentially plotting the ROC curve.

- **scikit-learn:**
  - A machine learning library providing various algorithms and tools.
  - **Used for:**
    - **Data splitting**: `train_test_split` to divide data into training and testing sets.
    - **Label encoding**: `LabelBinarizer` to convert text labels to numerical format.
    - **Metrics calculation**: `classification_report`, `roc_curve`, `auc` to evaluate model performance.

### Standard Libraries

- **os**: Provides functions for interacting with the operating system, such as handling file paths.
- **argparse**: Makes it easy to create command-line interfaces for your Python scripts.
- **time**: Provides time-related functions, such as pausing execution with `time.sleep`.
