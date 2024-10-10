# Face Mask Detection Using CNN and OpenCV

## Overview

This project aims to create a robust system for detecting face masks in images and real-time video streams. It leverages Convolutional Neural Networks (CNN) for accurate classification and uses the OpenCV library to implement live face detection capabilities.

## Features

- **Model Training:** Utilizes a CNN to classify faces as 'with mask' or 'without mask'.
- **Real-Time Detection:** Integrates with OpenCV to perform real-time mask detection using a live camera feed.
- **High Accuracy:** Achieves high accuracy on both training and test datasets.

## Dataset

The dataset comprises a balanced set of images with and without face masks to train the model effectively. It is split into training, testing and validation sets to ensure accurate validation of the model.

## Model Architecture

The model is built using a sequential CNN architecture with the following components:

- Convolutional layers with ReLU activation
- Max pooling for down-sampling
- Fully connected layers for classification
- An output layer with a softmax activation to distinguish between the two classes

## Installation and Setup

1. **Clone the Repository:**
   ```bash
   git clone <repository_link>
   cd FaceMaskDetection
   ```

2. **Install the Required Libraries:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Dataset:**
   Ensure the dataset is organized into the appropriate directories for training and testing.

4. **Train the Model:**
   Run the training script to train the model on the dataset.

5. **Real-Time Detection:**
   Use the following command to start the real-time detection:
   ```bash
   python detect_mask_video.py
   ```

## Usage

- The project can be used to monitor mask compliance in public spaces.
- Useful for deploying in environments where mask detection is necessary for health safety measures, such as airports, shopping malls, and offices.

## Conclusion

This project demonstrates a practical application of CNNs in computer vision, providing a tool that can assist in public health efforts to promote safety and compliance with health guidelines.

**Note:** For better accuracy in real-world scenarios, consider fine-tuning the model with more diverse data and adjusting the OpenCV face detection parameters to optimize performance.

## Acknowledgements

This project is inspired by the global need for safety during the COVID-19 pandemic, utilizing machine learning techniques to develop effective solutions.

## License

This project is licensed under the MIT License.

## References

1. CNN architecture and real-time detection information were adapted from various face mask detection projects using CNN.
