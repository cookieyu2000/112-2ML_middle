import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#設置資料路徑
train_data_dir = 'train'

#設置圖片尺寸
img_width, img_height = 224, 224

#設置批次大小
batch_size = 16

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

#凍結VGG16的權重
for layer in vgg16.layers:
    layer.trainable = False

#建立模型
model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(12, activation='softmax'))  # 12個類別

#編譯模型
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

#資料增強(Data Augmentation)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)  # 切分出20%的驗證集

#使用ImageDataGenerator從資料目錄中讀取圖片
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    seed=42)  # 使用訓練集

validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    seed=42)  # 使用驗證集

class_indices = train_generator.class_indices
print(class_indices)

#訓練模型
model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=100,  # 設置訓練的迭代次數
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size)

early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, mode='auto')
cheakpoint = ModelCheckpoint('vgg16_multiclass_model.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')

#保存模型
model.save('vgg16_multiclass_model.h5')