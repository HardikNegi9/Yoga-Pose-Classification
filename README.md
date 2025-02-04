# 🧘 Yoga Pose Classification

This project classifies yoga poses out of 107 different yoga poses using DenseNet121.

## 📂 Dataset
The dataset used for this project can be found on Kaggle: [Yoga Pose Image Classification Dataset](https://www.kaggle.com/datasets/shrutisaxena/yoga-pose-image-classification-dataset).

## ⚙️ Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/HardikNegi9/Yoga-Pose-Classification.git
    cd Yoga-Pose-Classification
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## 🚀 Usage
1. To train the model, run the Jupyter notebook:
    - `ModelTraining.ipynb`

2. To predict a yoga pose from an image, run the script:
    ```sh
    python main.py
    ```

## 🏋️ Model Training
The model is built using TensorFlow and Keras, leveraging the DenseNet121 architecture. The model is trained on the yoga pose dataset and saved as `yoga_pose_model.h5`.

### 🔄 Preprocessing
Images are resized to 100x100 pixels and normalized. Labels are converted to categorical format using `to_categorical`.

### 📈 Data Augmentation
Performed using `ImageDataGenerator`:
```python
datagen = ImageDataGenerator(horizontal_flip=False,
                             vertical_flip=False,
                             rotation_range=0,
                             zoom_range=0.2,
                             width_shift_range=0,
                             height_shift_range=0,
                             shear_range=0,
                             fill_mode="nearest")
```

### 🏗️ Model Architecture
Utilizing DenseNet121 as the base model with additional layers:
```python
pretrained_model = tf.keras.applications.DenseNet121(input_shape=(100,100,3),
                                                      include_top=False,
                                                      weights='imagenet',
                                                      pooling='avg')
pretrained_model.trainable = False

inputs = pretrained_model.input
drop_layer = tf.keras.layers.Dropout(0.25)(pretrained_model.output)
x_layer = tf.keras.layers.Dense(512, activation='relu')(drop_layer)
x_layer1 = tf.keras.layers.Dense(128, activation='relu')(x_layer)
drop_layer1 = tf.keras.layers.Dropout(0.20)(x_layer1)
outputs = tf.keras.layers.Dense(107, activation='softmax')(drop_layer1)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
```

### ⚙️ Training
Compiled and trained with:
```python
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(datagen.flow(X_train, Y_train, batch_size=32), validation_data=(X_val, Y_val), epochs=20)
```

### 💾 Saving the Model
```python
model.save("yoga_pose_model.h5")
```

## 📊 Results
The training and validation accuracy and loss are plotted in `ModelTraining.ipynb`.

## 🖥️ GUI for Prediction
A simple GUI is provided in `main.py` to select an image and predict the yoga pose using Tkinter.
