# Dog vs Cat Image Classifier (CNN with Keras)

This project is a Convolutional Neural Network (CNN) built using Keras to classify images of cats and dogs. It is inspired by the classic Kaggle dataset challenge.

## 🔍 Project Overview

- **Goal**: Classify input images as either a dog or a cat.
- **Framework**: Keras with TensorFlow backend.
- **Platform**: Developed and trained on Google Colab.

## ⚠ Resource Limitations

Due to **RAM**, **GPU**, and **storage constraints** in Google Colab:

- We only used **1,000 cat** and **1,000 dog** images for training.
- We used **data generators** (`ImageDataGenerator`) to batch the data into **20-image chunks** during training, preventing RAM overload.

## 🧠 Model Architecture

- 4 convolutional layers with max-pooling
- Dropout layer (optional) to prevent overfitting
- Dense output layer with sigmoid activation
- Optimizer: RMSprop
- Loss function: Binary Crossentropy

```python
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))  # Only for dropout variant
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```

## 🖼 Dataset Details

- **Training Set**: 1000 cat + 1000 dog images
- **Validation Set**: 1000 cat + 1000 dog images
- **Test Set**: 1000 mixed unlabeled images

## ⚙ Training Comparison

| Model Variant     | Epochs | Train Accuracy | Val Accuracy | Train Loss | Val Loss |
|------------------|--------|----------------|--------------|------------|----------|
| Without Dropout  | 30     | 0.9897         | 0.7425       | 0.0579     | 0.8293   |
| With Dropout     | 100    | 0.9937         | 0.7785       | 0.0199     | 1.0745   |

💡 **Observation**: Dropout helped increase validation accuracy and reduced overfitting, though the validation loss increased slightly.

## 📁 Folder Structure

```
base_dir/
├── train/
│   ├── cats/
│   └── dogs/
├── validation/
│   ├── cats/
│   └── dogs/
└── test/
```

## 💾 Model Saving

The final trained model is saved in Keras `.keras` format:
```python
model.save("cats_and_dogs_small_1.keras")
```

## ☁ GitHub LFS

The `.keras` model is tracked using **Git Large File Storage (Git LFS)** due to its size exceeding GitHub’s 25MB limit.

## 📦 Requirements

- Python 3.x
- TensorFlow / Keras
- Matplotlib
- Git LFS

## 📈 Future Work

- Train on full dataset
- Introduce transfer learning (e.g., VGG, ResNet)
- Export model to TFLite for mobile deployment

## 📬 Contact

For any feedback or questions, feel free to reach out via GitHub Issues.
