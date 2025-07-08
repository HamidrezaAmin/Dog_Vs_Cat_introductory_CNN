 Dog vs Cat Image Classifier (CNN with Keras)

This project is a Convolutional Neural Network (CNN) built using Keras to classify images of cats and dogs. It is inspired by the classic Kaggle dataset challenge.

 🔍 Project Overview

- **Goal**: Classify input images as either a dog or a cat.
- **Framework**: Keras with TensorFlow backend.
- **Platform**: Developed and trained on Google Colab.

 🧠 Model Architecture

- 4 convolutional layers with max-pooling
- Dropout layer to prevent overfitting
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
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```

 🖼 Dataset Details

Due to Google Colab limitations (RAM, disk size, GPU quota), only a **subset** of the dataset was used:

- **Training Set**:
  - 1000 cat images
  - 1000 dog images

- **Validation Set**:
  - 1000 cat images
  - 1000 dog images (next 1000 from dataset)

- **Test Set**:
  - 1000 mixed images (no labels used)

 🧪 Evaluation

Validation accuracy reached ~88% while monitoring overfitting using dropout and augmentation.

Accuracy and loss plots were generated to analyze training behavior.

 📁 Folder Structure

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

 💾 Model Saving

The final trained model is saved in Keras `.keras` format:
```python
model.save("cats_and_dogs_small_1.keras")
```

 ☁ GitHub LFS

Because the `.keras` model is over 25MB, it is uploaded using **Git Large File Storage (Git LFS)**. Make sure you have Git LFS installed to clone this repository.

 📦 Requirements

- Python 3.x
- TensorFlow / Keras
- Matplotlib
- Git LFS (for handling large model files)

 📈 Future Work

- Scale to full dataset
- Try different architectures (ResNet, VGG)
- Apply transfer learning

 📬 Contact

For any feedback or questions, feel free to reach out via GitHub Issues.
