# Grape Leaf Classification Using Pre-trained Deep Learning Models

This project presents code and resources for classifying grapevine leaves into distinct categories. The primary objective is to develop a robust machine learning model capable of accurately identifying various types of grapevine leaves, which can significantly benefit viticulture and botanical research.

## Dataset
The dataset utilized in this project comprises images of grapevine leaves, each labeled according to its respective leaf type. The dataset includes the following classes:
- Ak
- Ala_Idris
- Buzgulu
- Dimnit
- Nazli

## Requirements
To execute this project, you will need the following libraries and dependencies:
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- Pandas

## Data Preprocessing
The dataset undergoes processing and is divided into training, validation, and test sets. Data augmentation techniques are applied to the training set to enhance the diversity of images, which helps improve the model's generalization capabilities.

### Key Preprocessing Functions:
- **load_dataset()**: Loads the dataset from the specified path.
- **preprocess_images()**: Applies preprocessing steps, including resizing and normalization of the images.
- **split_dataset()**: Segments the dataset into training, validation, and test sets.

## Model Architectures
Multiple pre-trained deep learning models are employed for this classification task, including:
- VGG19
- ResNet50
- Inception-V3
- EfficientNet B3

Each model is fine-tuned for the specific task by adjusting the output layers and retraining on the grapevine leaf dataset.

### Essential Model Functions:
1. **build_model()**: Constructs the desired deep learning model architecture.
2. **fine_tune_model()**: Fine-tunes the pre-trained model on the grapevine leaf dataset.

## Training
The models are trained using the training set and evaluated on both the validation and test sets. Various hyperparameters, such as learning rates and the number of epochs, are optimized to enhance model performance.

### Key Training Functions:
- **train_model()**: Conducts the training of the model on the training dataset.
- **evaluate_model()**: Assesses the model's performance on the validation and test datasets.
- **hyperparameter_tuning()**: Adjusts hyperparameters for optimal performance.

