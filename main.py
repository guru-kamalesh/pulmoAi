import tensorflow as tf
import os
import json

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# ==============================
# CREATE MODEL FOLDER AUTOMATICALLY
# ==============================
os.makedirs("model", exist_ok=True)

DATASET_PATH = "dataset"

# ==============================
# DATA PREPROCESSING
# ==============================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

train = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# ==============================
# SAVE CLASS LABELS
# ==============================
with open("model/class_indices.json", "w") as f:
    json.dump(train.class_indices, f)

print("Class Labels:", train.class_indices)

# ==============================
# MODEL BUILDING (TRANSFER LEARNING)
# ==============================
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model
for layer in base_model.layers:
    layer.trainable = False

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==============================
# TRAINING
# ==============================
history = model.fit(
    train,
    validation_data=val,
    epochs=5
)

# ==============================
# SAVE MODEL
# ==============================
model.save("model/model.h5")

print("✅ MODEL TRAINING COMPLETED & SAVED")