import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

from tensorflow.keras.applications import MobileNetV2

# Dataset paths
with_mask_path = 'dataset/with_mask'
without_mask_path = 'dataset/without_mask'

categories = [with_mask_path, without_mask_path]

labels = [0, 1]

data = []
target = []

# Read images
for category, label in zip(categories, labels):

    for image_name in os.listdir(category):

        image_path = os.path.join(category, image_name)

    img = cv2.imread(image_path)

    try:
            img = cv2.resize(img, (100, 100))
            data.append(img)
            target.append(label)

    except:
            pass

# Convert to numpy
X = np.array(data)
X = X / 255.0

y = np.array(target)
y = to_categorical(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
     #CNN Model
model = Sequential()

model.add(
    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(100,100,3)
    )
)

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(
    Conv2D(
        64,
        (3,3),
        activation='relu'
    )
)

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_test, y_test)
)

# Save model
model.save('face_mask_model.h5')

print('Model Trained Successfully')

# Accuracy Graph
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.legend(['Training', 'Validation'])

plt.show()