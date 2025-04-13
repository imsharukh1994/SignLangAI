import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# --- Configuration ---
dataset_path = 'data/train'     # Path where your A-Z folders are
img_size = 64                   # Resize images to 64x64
batch_size = 64
epochs = 10
model_output = 'model/ASL_model.h5'

# --- Load Dataset ---
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# --- Model ---
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- Train Model ---
if not os.path.exists('model'):
    os.makedirs('model')

checkpoint = ModelCheckpoint(model_output, save_best_only=True, monitor='val_accuracy', mode='max')

model.fit(train_data, validation_data=val_data, epochs=epochs, callbacks=[checkpoint])

print(f"\n✅ Model training complete. Saved to: {model_output}")
