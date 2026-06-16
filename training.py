import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

# =========================
# 1. SETUP DATASET
# =========================
base_dir = 'dataset/Oily-Dry-Skin-Types'
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'valid')

# =========================
# 2. DATA AUGMENTATION (Biar AI tidak gampang bingung)
# =========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical'
)

valid_generator = val_datagen.flow_from_directory(
    valid_dir, target_size=(224, 224), batch_size=32, class_mode='categorical'
)

# =========================
# 3. TRANSFER LEARNING (MobileNetV2)
# =========================
base_model = applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights='imagenet'
)
base_model.trainable = False  # Membekukan bobot awal agar stabil

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(train_generator.class_indices), activation='softmax')
])

# =========================
# 4. TRAINING
# =========================
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Callback untuk menurunkan learning rate jika val_loss tidak membaik
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-6, verbose=1)

# Early stopping agar training berhenti jika tidak ada peningkatan
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


EPOCHS = 15
print(f"Memulai training cerdas dengan Transfer Learning selama {EPOCHS} epoch...")

# Tahap 1: Training dengan base_model dibekukan
history = model.fit(
    train_generator,
    epochs=EPOCHS//2,
    validation_data=valid_generator,
    callbacks=[early_stop, reduce_lr]
)

# Tahap 2: Fine-tuning beberapa layer akhir base_model
print("Membuka 30 layer terakhir MobileNetV2 untuk fine-tuning...")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

history_finetune = model.fit(
    train_generator,
    epochs=EPOCHS - (EPOCHS//2),
    validation_data=valid_generator,
    callbacks=[early_stop, reduce_lr]
)

# Gabungkan history
for key in history.history:
    history.history[key] += history_finetune.history[key]

# =========================
# 5. SIMPAN MODEL
# =========================
model.save('skin_model.h5')
print("✅ Training selesai! Model 'skin_model.h5' siap digunakan.")

# =========================
# 6. GENERATE & SIMPAN GRAFIK PERFORMA
# =========================
print("Sedang membuat grafik evaluasi...")

# Mengambil data performa dari variabel history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

# 1. Membuat & Menyimpan Grafik Akurasi
plt.figure(figsize=(10, 5))
plt.plot(epochs_range, acc, label='Training Accuracy', color='#1f77b4', linewidth=2)
plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='#ff7f0e', linewidth=2)
plt.title('Grafik Akurasi Model (MobileNetV2)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('akurasi.png', dpi=300) # Menyimpan file resolusi tinggi
plt.close() # Menutup window plotting agar hemat memori
print("✅ Grafik Akurasi sukses disimpan sebagai 'akurasi.png'")

# 2. Membuat & Menyimpan Grafik Loss
plt.figure(figsize=(10, 5))
plt.plot(epochs_range, loss, label='Training Loss', color='#d62728', linewidth=2)
plt.plot(epochs_range, val_loss, label='Validation Loss', color='#2ca02c', linewidth=2)
plt.title('Grafik Loss Model (MobileNetV2)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('loss.png', dpi=300) # Menyimpan file resolusi tinggi
plt.close()
print("✅ Grafik Loss sukses disimpan sebagai 'loss.png'")