# Grapevine Leaf Classification Using Pre-trained Deep Learning Models

This project presents the classification of grapevine leaves into distinct categories using deep learning models. The goal is to develop a machine learning model that accurately identifies various types of grapevine leaves, which can significantly benefit applications like viticulture, disease detection, and yield prediction in agriculture.


## Dataset <a name="dataset"></a>
The dataset consists of images of grapevine leaves categorized into distinct classes based on their varieties. The following leaf types are included in the dataset:
- Ak
- Ala_Idris
- Buzgulu
- Dimnit
- Nazli

---

## Data Preparation 

### Data Collection
The dataset contains images of grapevine leaves from different varieties, with labels indicating the respective leaf types. These images are organized into appropriate categories.

### Data Preprocessing
- Images are resized to a uniform size (e.g., 227x227 pixels).
- Data augmentation techniques such as rotation, brightness adjustment, and horizontal flipping are applied to enhance diversity and improve model generalization.

- The dataset is split into three parts: training, validation, and test sets.

### Key Preprocessing Functions
1. **`load_dataset()`**: Loads the dataset from the specified path.
2. **`preprocess_images()`**: Applies preprocessing steps including resizing and normalization.
3. **`split_dataset()`**: Divides the dataset into training, validation, and test sets.

---

## Model Architectures 
Several pre-trained deep learning models are used in this classification task. Fine-tuning is applied to the following models to adjust their output layers and retrain them on the grapevine leaf dataset:
- **EfficientNet B3**
- **ResNet50**
- **VGG19**
- **Inception-V3**
- **Xception**

### Essential Model Functions
1. **`build_model()`**: Constructs the model architecture.
2. **`fine_tune_model()`**: Fine-tunes the pre-trained model to fit the grapevine leaf dataset.

---

## Model Training 
Each model is trained on the training dataset and validated using the validation dataset. Various hyperparameters, such as learning rates and the number of epochs, are optimized to achieve better performance.

### Key Training Functions
1. **`train_model()`**: Executes the training process on the training dataset.
2. **`evaluate_model()`**: Evaluates the model’s performance on the validation and test sets.
3. **`hyperparameter_tuning()`**: Tunes hyperparameters like learning rates and epochs.

---

## Model Evaluation 
After training, the models are evaluated on the test dataset. The primary evaluation metrics used are:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

A confusion matrix is generated for each model to provide insight into the classification results and show misclassifications.
