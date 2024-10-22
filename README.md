# Grapevine Leaves Classification Using Pre-trained Deep Learning Models

This project focuses on classifying grapevine leaves into different categories using data mining techniques and deep learning models. The primary objective is to develop a robust machine learning model capable of accurately identifying various types of grapevine leaves, significantly benefiting agriculture through applications such as disease detection and yield prediction.

## Introduction <a name="introduction"></a>

The objective of this project is to classify grapevine leaves into different categories. This classification task is essential for various applications in agriculture, such as disease detection and yield prediction. The project utilizes machine learning and deep learning techniques to achieve accurate classification results.

## Data Preparation <a name="data-preparation"></a>

### Data Collection <a name="data-collection"></a>

The dataset used in this project contains images of grapevine leaves from different varieties. The images are organized into categories based on the type of leaves.

### Data Preprocessing <a name="data-preprocessing"></a>

- The images are resized to a consistent size (e.g., 227x227 pixels) to ensure uniformity.
- Data augmentation techniques such as rotation, brightness adjustment, and horizontal flipping are applied to increase the dataset's diversity.

- The dataset is split into training, validation, and test sets.

## Model Architectures <a name="model-architectures"></a>

Several deep learning models are employed for the classification task, including:
- **EfficientNetB3**
- **ResNet50**
- **VGG19**
- **Xception**

Each model is fine-tuned for the specific task by adjusting the output layers and retraining on the grapevine leaf dataset.

## Model Training <a name="model-training"></a>

The models are trained on the training dataset, and their performance is evaluated on the validation set. Various hyperparameters, such as learning rates and the number of epochs, are optimized to enhance model performance.

### Key Training Functions
- **train_model()**: Conducts training on the training dataset.
- **evaluate_model()**: Assesses performance on the validation dataset.
- **hyperparameter_tuning()**: Adjusts hyperparameters for optimal performance.

## Model Evaluation <a name="model-evaluation"></a>

The selected model's performance is assessed on the test dataset. Evaluation metrics include accuracy, precision, recall, and F1-score. A confusion matrix is also generated to visualize the classification results.

## Results <a name="results"></a>

The evaluation results for each model, including accuracy and confusion matrices, are summarized below.

### Model Performance on Validation Set

| Model           | Accuracy | Precision | Recall | F1 Score |
|-----------------|----------|-----------|--------|----------|
| VGG19           | 0.92     | 0.91      | 0.93   | 0.92     |
| ResNet50        | 0.94     | 0.94      | 0.94   | 0.94     |
| Inception-V3    | 0.91     | 0.92      | 0.90   | 0.91     |
| EfficientNet B3 | 0.95     | 0.95      | 0.95   | 0.95     |
| Xception        | TBD      | TBD       | TBD    | TBD      |

### Model Performance on Test Set

| Model           | Accuracy | Precision | Recall | F1 Score |
|-----------------|----------|-----------|--------|----------|
| VGG19           | 0.91     | 0.90      | 0.92   | 0.91     |
| ResNet50        | 0.93     | 0.93      | 0.93   | 0.93     |
| Inception-V3    | 0.90     | 0.91      | 0.89   | 0.90     |
| EfficientNet B3 | 0.94     | 0.94      | 0.94   | 0.94     |
| Xception        | TBD      | TBD       | TBD    | TBD      |

Additionally, a **10-fold cross-validation** approach is employed for a comprehensive assessment of each model's performance.

## Conclusion <a name="conclusion"></a>

This project demonstrates the effectiveness of machine learning models in classifying grapevine leaves. The model with the highest accuracy and reliability can be selected for practical applications in viticulture.
