import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# 載入模型
model = load_model('VGG16_Best3.h5')

# 定義圖片尺寸
img_width, img_height = 224, 224

# 測試資料夾路徑
test_folder = 'test'

# 定義類別名稱和對應的索引值
class_labels = {'Bean': 0, 'Broccoli': 1, 'Cabbage': 2, 'Capsicum': 3,
                'Carrot': 4, 'Cauliflower': 5, 'Cucumber': 6, 'Potato': 7, 'Pumpkin': 8,
                'Radish': 9, 'Tomato': 10}

# 初始化計數器
total_images = 0
correct_predictions = 0

labels=[]
pre=[]

# 迭代每個類別名稱
for class_name, class_index in class_labels.items():
    # 取得分類資料夾的路徑
    class_folder_path = os.path.join(test_folder, class_name)

    # 檢查路徑是否存在
    if not os.path.exists(class_folder_path):
        continue

    # 取得分類資料夾中所有圖片的路徑
    images = os.listdir(class_folder_path)

    # 迭代每個圖片
    for img_name in images:
        # 圖片路徑
        img_path = os.path.join(class_folder_path, img_name)

        # 載入圖片並調整大小
        img = image.load_img(img_path, target_size=(img_width, img_height))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        labels.append(class_index)

        # 預測圖片類別
        prediction = model.predict(img)

        # 取得預測結果的索引值
        predicted_class_index = np.argmax(prediction)
        pre.append(predicted_class_index)

        # 檢查預測結果是否正確
        if predicted_class_index == class_index:
            correct_predictions += 1
            # plt.imshow(img[0])
            # plt.title(class_index)
            # plt.axis('off')
            # plt.show()

        total_images += 1


#計算準確率
accuracy = correct_predictions / total_images

# 輸出準確率
print("Accuracy: {:.2f}%".format(accuracy * 100))

# # 計算準確度
# accuracy = accuracy_score(labels, pre)

# # 輸出準確度
# print(f"預測準確度: {accuracy:.4f}")