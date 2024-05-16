from keras.preprocessing.image import ImageDataGenerator
import os
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.utils import load_img, img_to_array
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

label_num={
    0: 'Bean',
    1: 'Broccoli',
    2: 'Cabbage',
    3: 'Capsicum',
    4: 'Carrot',
    5: 'Cauliflower',
    6: 'Cucumber',
    7: 'Potato',
    8: 'Pumpkin',
    9: 'Radish',
    10: 'Tomato'
}


model_path='VGG16_Best3.h5'
model=load_model(model_path)
test_data_dir='train'
# 定義 ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1./255)  # 根據需要設置預處理參數

# 從目錄中讀取測試集
test_generator = test_datagen.flow_from_directory(
    test_data_dir,  # 測試數據目錄的路徑
    target_size=(224, 224),  # 圖像目標大小
    batch_size=4,  # 批次大小
    class_mode='categorical', # 類型模式
    seed=42 
)

# 使用模型進行預測
predictions = model.predict(test_generator)
predicted_labels=predictions.argmax(axis=1)

# 獲取真實標籤
true_labels = []
for images, labels in test_generator:
    true_labels.extend(labels.argmax(axis=1))

    # plt.imshow(images[0])
    # plt.title(label_num[np.argmax(labels[0])])
    # plt.axis('off')
    # plt.show()
    # 您可以根據需要決定是否提前停止迭代
    if len(true_labels) == len(predicted_labels):
        break

# 計算準確度
accuracy = accuracy_score(true_labels, predicted_labels)

# 輸出準確度
print(f"預測準確度: {accuracy:.4f}")

print(model.summary())