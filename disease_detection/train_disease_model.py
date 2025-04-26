# disease_detection/train_disease_model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                      ReduceLROnPlateau, TensorBoard)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report

# Configuration
TARGET_SIZE = (300, 300)  # Increased for EfficientNet
BATCH_SIZE = 64
EPOCHS = 30
BASE_LEARNING_RATE = 1e-4
FINE_TUNE_LEARNING_RATE = 1e-6
AUTOTUNE = tf.data.AUTOTUNE

# Enable mixed precision training
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Advanced data augmentation with EfficientNet preprocessing
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    validation_split=0.15
)

test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    validation_split=0.15
)

# Create data pipelines
def create_data_pipeline(generator, subset):
    return generator.flow_from_directory(
        'plant_disease_dataset/',
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset=subset,
        shuffle=(subset == 'training'),
        seed=42
    )

train_generator = create_data_pipeline(train_datagen, 'training')
val_generator = create_data_pipeline(test_datagen, 'validation')
test_generator = create_data_pipeline(test_datagen, 'validation')  # Adjust split as needed

# Handle class imbalance
class_weights = {i: 1.0 / count for i, count in enumerate(train_generator.classes)}
total_samples = sum(train_generator.classes)
class_weights = {k: v * total_samples for k, v in class_weights.items()}

# Build advanced model using EfficientNet
def build_model(num_classes):
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=TARGET_SIZE + (3,)
    )
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

model = build_model(train_generator.num_classes)

# Callbacks
callbacks = [
    ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss'),
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.1, patience=2),
    TensorBoard(log_dir='./logs')
]

# Training in two phases: feature extraction and fine-tuning

# Phase 1: Feature extraction (frozen base)
print("Phase 1: Feature extraction")
base_model = model.layers[0]
base_model.trainable = False

model.compile(optimizer=Adam(BASE_LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC()])

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=2
)

# Phase 2: Fine-tuning (partial unfreezing)
print("\nPhase 2: Fine-tuning")
base_model.trainable = True
for layer in base_model.layers[:-10]:
    layer.trainable = False

model.compile(optimizer=Adam(FINE_TUNE_LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC()])

history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS + 10,
    initial_epoch=history.epoch[-1],
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=2
)

# Final evaluation
print("\nFinal evaluation:")
model.load_weights('best_model.h5')  # Load best weights
test_loss, test_acc, test_auc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.2%}, AUC: {test_auc:.3}")

# Save final model
model.save('plant_disease_cnn_advanced.keras')

# Generate classification report
y_true = test_generator.classes
y_pred = model.predict(test_generator).argmax(axis=1)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=list(train_generator.class_indices.keys())))

# Inference with class mapping
class_indices = train_generator.class_indices
class_names = {v: k for k, v in class_indices.items()}

def predict_disease(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=TARGET_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    
    predictions = model.predict(img_array)
    scores = {class_names[i]: float(pred) for i, pred in enumerate(predictions[0])}
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    print("Diagnosis Results:")
    for class_name, score in sorted_scores[:3]:
        print(f"- {class_name}: {score:.2%} confidence")
    return sorted_scores[0][0]

# Example usage
print(predict_disease('sick_plant.jpg'))