"""
Train CNN model using HAM10000 dataset for skin cancer detection
Dataset: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

Instructions:
1. Download HAM10000 dataset from Kaggle
2. Extract to a folder (e.g., C:/datasets/HAM10000/)
3. Update DATASET_PATH below
4. Run: python train_model_ham10000.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# ============================================================================
# CONFIGURATION
# ============================================================================
DATASET_PATH = "C:/Users/MSI 123/Downloads/archive (2)/"  # HAM10000 dataset path
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001

# For binary classification (benign vs malignant)
# Malignant: mel (melanoma), bcc (basal cell carcinoma)
# Benign: nv, bkl, df, vasc, akiec
BINARY_CLASSIFICATION = True

# ============================================================================
# LOAD DATASET
# ============================================================================
print("Loading HAM10000 metadata...")
metadata_path = os.path.join(DATASET_PATH, "HAM10000_metadata.csv")
df = pd.read_csv(metadata_path)

print(f"Total images: {len(df)}")
print(f"\nClass distribution:")
print(df['dx'].value_counts())

# ============================================================================
# BINARY CLASSIFICATION MAPPING
# ============================================================================
if BINARY_CLASSIFICATION:
    # Map to binary: 0=benign, 1=malignant
    malignant_classes = ['mel', 'bcc']
    df['binary_label'] = df['dx'].apply(lambda x: 1 if x in malignant_classes else 0)
    
    print(f"\nBinary classification:")
    print(f"Benign: {(df['binary_label'] == 0).sum()}")
    print(f"Malignant: {(df['binary_label'] == 1).sum()}")
    
    label_column = 'binary_label'
    num_classes = 1  # Binary classification
else:
    # Multi-class classification (7 classes)
    label_map = {
        'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 
        'akiec': 4, 'vasc': 5, 'df': 6
    }
    df['label'] = df['dx'].map(label_map)
    label_column = 'label'
    num_classes = 7

# ============================================================================
# CREATE IMAGE PATHS
# ============================================================================
# HAM10000 images are in two folders
image_folder_1 = os.path.join(DATASET_PATH, "HAM10000_images_part_1")
image_folder_2 = os.path.join(DATASET_PATH, "HAM10000_images_part_2")

def get_image_path(image_id):
    """Find image path in either folder"""
    path1 = os.path.join(image_folder_1, f"{image_id}.jpg")
    path2 = os.path.join(image_folder_2, f"{image_id}.jpg")
    if os.path.exists(path1):
        return path1
    elif os.path.exists(path2):
        return path2
    else:
        return None

df['image_path'] = df['image_id'].apply(get_image_path)
df = df[df['image_path'].notna()]  # Remove missing images

print(f"\nImages found: {len(df)}")

# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================
train_df, test_df = train_test_split(
    df, 
    test_size=0.2, 
    stratify=df[label_column],
    random_state=42
)

print(f"\nTrain samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")

# ============================================================================
# DATA AUGMENTATION
# ============================================================================
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2],
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)

test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)

# ============================================================================
# DATA GENERATORS
# ============================================================================
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='image_path',
    y_col=label_column,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='raw' if BINARY_CLASSIFICATION else 'categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col='image_path',
    y_col=label_column,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='raw' if BINARY_CLASSIFICATION else 'categorical',
    shuffle=False
)

# ============================================================================
# COMPUTE CLASS WEIGHTS (for imbalanced data)
# ============================================================================
class_weights_array = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_df[label_column]),
    y=train_df[label_column]
)
class_weights_dict = dict(enumerate(class_weights_array))
print(f"\nClass weights: {class_weights_dict}")

# ============================================================================
# BUILD MODEL (Transfer Learning with EfficientNetB0)
# ============================================================================
print("\nBuilding model with EfficientNetB0...")

base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model initially
base_model.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

if BINARY_CLASSIFICATION:
    output = Dense(1, activation='sigmoid')(x)
    loss = 'binary_crossentropy'
    metrics = ['accuracy', tf.keras.metrics.AUC(name='auc'), 
               tf.keras.metrics.Precision(name='precision'),
               tf.keras.metrics.Recall(name='recall')]
else:
    output = Dense(num_classes, activation='softmax')(x)
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss=loss,
    metrics=metrics
)

print(model.summary())

# ============================================================================
# CALLBACKS
# ============================================================================
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'skin_cancer_cnn_best.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# ============================================================================
# TRAIN MODEL (Phase 1: Frozen base)
# ============================================================================
print("\n" + "="*60)
print("PHASE 1: Training with frozen base model")
print("="*60)

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_df) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=test_generator,
    validation_steps=len(test_df) // BATCH_SIZE,
    callbacks=[early_stopping, lr_scheduler, checkpoint],
    class_weight=class_weights_dict
)

# ============================================================================
# FINE-TUNING (Phase 2: Unfreeze last layers)
# ============================================================================
print("\n" + "="*60)
print("PHASE 2: Fine-tuning (unfreezing last 20 layers)")
print("="*60)

base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE / 10),  # Lower learning rate
    loss=loss,
    metrics=metrics
)

history_fine = model.fit(
    train_generator,
    steps_per_epoch=len(train_df) // BATCH_SIZE,
    epochs=20,
    validation_data=test_generator,
    validation_steps=len(test_df) // BATCH_SIZE,
    callbacks=[early_stopping, lr_scheduler, checkpoint],
    class_weight=class_weights_dict
)

# ============================================================================
# SAVE FINAL MODEL
# ============================================================================
model.save('skin_cancer_cnn.h5')
print("\n✅ Model saved as: skin_cancer_cnn.h5")

# ============================================================================
# EVALUATE MODEL
# ============================================================================
print("\n" + "="*60)
print("FINAL EVALUATION")
print("="*60)

results = model.evaluate(test_generator, steps=len(test_df) // BATCH_SIZE)
print(f"\nTest Loss: {results[0]:.4f}")
print(f"Test Accuracy: {results[1]:.4f}")
if BINARY_CLASSIFICATION:
    print(f"Test AUC: {results[2]:.4f}")
    print(f"Test Precision: {results[3]:.4f}")
    print(f"Test Recall: {results[4]:.4f}")

# ============================================================================
# PLOT TRAINING HISTORY
# ============================================================================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
print("\n📊 Training history saved as: training_history.png")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"Final model: skin_cancer_cnn.h5")
print(f"Best model: skin_cancer_cnn_best.h5")
print(f"Copy the best model to your backend folder to use it.")
