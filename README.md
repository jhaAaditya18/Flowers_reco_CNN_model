# Flowers_reco_CNN_model
basic flowers recorgination model of cnn project

# 🌸 Flower Classification Using CNN

This project implements a Convolutional Neural Network (CNN) in TensorFlow/Keras to classify images of five types of flowers: Daisy, Dandelion, Rose, Sunflower, and Tulip.

---

## 📁 Dataset Overview

- **Path**: `C:/Users/ajha7/OneDrive/Desktop/Cnn project/img`
- **Classes**: `daisy`, `dandelion`, `rose`, `sunflower`, `tulip`
- **Total Images**: 4,317  
  - daisy: 764  
  - dandelion: 1,052  
  - rose: 784  
  - sunflower: 733  
  - tulip: 984

---

## 🧰 Libraries & Tools

- Python 3.x
- TensorFlow / Keras
- NumPy, Matplotlib
- PIL (via TensorFlow utilities)

---

## 🔧 Data Pipeline

1. **Image Loading**: Using `image_dataset_from_directory()`
2. **Splitting**:  
   - 80% for training  
   - 20% for validation
3. **Image Size**: 180×180 pixels
4. **Batch Size**: 32
5. **Label Format**: Integer encoding

---

## 🧠 Model Architecture
```text
Input Layer (180x180x3)
↓
Data Augmentation (RandomFlip, RandomRotation, RandomZoom)
↓
Rescaling (1./255)
↓
Conv2D → ReLU → MaxPooling
↓
Conv2D → ReLU → MaxPooling
↓
Conv2D → ReLU → MaxPooling
↓
Dropout (0.2)
↓
Flatten
↓
Dense(128) → ReLU
↓
Dense(5) → Softmax

```
⚙️ Compilation
Loss Function: SparseCategoricalCrossentropy

Optimizer: Adam

Metrics: Accuracy

📈 Model Training
Epochs: 10

Best Accuracy:

Training: ~75.4%

Validation: ~70.2%

Loss decreased steadily over time.

📊 Visualizations
9 sample training images shown with their class labels.

Augmented image samples shown to verify augmentation pipeline.

🧪 Inference Function
python
Copy
Edit
def classify_images(image_path):
    input_images = tf.keras.utils.load_img(image_path, target_size=(180,180))
    input_images_array = tf.keras.utils.img_to_array(input_images)
    input_images_exp_dim = tf.expand_dims(input_images_array, 0)

    predictions = model.predict(input_images_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = f"The flower belongs to {flower_names[np.argmax(result)]} with a score of {round(np.max(result)*100, 2)}%"
    return outcome
⚠️ Bug Fixes:

Corrected the use of load_img from tf.keras.utils (was previously a typo).

Replaced np.arg() with np.argmax().

🛠️ Future Improvements
Add model checkpointing and early stopping.

Increase training epochs.

Improve generalization with more complex augmentation and layers.

Export model to ONNX or TFLite for mobile applications.

📬 Author Info
Author: Aaditya Jha
Institute: KPR Institute of Engineering and Technology
Project: Flower Classifier with CNN - Mini Project (2025)

📌 Acknowledgements
TensorFlow and Keras documentation.

Public datasets of flower images.

Academic guidance from faculty.

📸 Sample Prediction
python
Copy
Edit
classify_images("sample/sunflower.jpg")
# Output: "The flower belongs to sunflower with a score of 87.53%"
yaml
Copy
Edit
