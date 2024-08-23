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

You can install the required Python packages using the following command:

```bash
pip install -r requirements.txt
