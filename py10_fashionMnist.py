

# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 14:22:55 2026

@author: letuin
"""


from tensorflow.keras.datasets.fashion_mnist import load_data  #데이터 수집
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

def makeImg(idx):
    tempimg = x_test[idx]
    timg=Image.fromarray(tempimg)
    timg.save(f'image/fashion_{idx}.jpg', 'jpeg')



# %% 

# 1) 데이터 수집 
(Tx_train, Ty_train), (x_test, y_test)=load_data()

makeImg(30)


# 2) 데이터 전처리
# 2-1) val 자료 분리
# from sklearn.model_selection import train_test_split
# test_size=0.2,  :  8:2
x_train, x_val, y_train, y_val = \
  train_test_split(Tx_train, Ty_train, test_size=0.2, 
                              random_state=777)   
  
# 2-2) x 정규화 (normalization)
x_train=x_train.reshape(48000, 28*28)/255  # 2dim
x_val=x_val.reshape(12000, 28*28)/255
x_test=x_test.reshape(10000, 28*28)/255


# 2-3 one-hot encoding
y_train=to_categorical(y_train)
y_val=to_categorical(y_val)
y_test=to_categorical(y_test)


# 3. model 정의
model=Sequential([
    Dense(units=64, activation='relu', input_shape=(784,)),
    Dense(units=32, activation='relu'),
    Dense(units=10, activation='softmax')   
    ])

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=['acc'])

history = model.fit(x_train, y_train,
          epochs=30,
          batch_size=127,
          validation_data=(x_val, y_val)          
          )


# 시각화
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['acc'], 'b-', label='accuracy')
plt.plot(history.history['val_acc'], 'r--', label='val_accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()  
  

# 4. 평가
model.evaluate(x_test, y_test)

# 5. 예측
results = model.predict(x_test)

results[1]
y_test[1]

y_pred=np.argmax(results, axis=1)
y_true=np.argmax(y_test, axis=1)

# heatmap
cm = confusion_matrix(y_true, y_pred)
cm

sns.heatmap(cm, fmt='g', annot=True, cmap='Blues')
plt.title('Confusion Matrix', fontsize=14)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.show()

class_names=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#예측이 틀린 이미지 16개 프린트 
count=1
for idx  in range(len(y_true)):
    if y_pred[idx] != y_true[idx]:
        plt.subplot(4,4, count)
        count +=1
        plt.imshow(x_test[idx].reshape(28,28))
        plt.title('pred:%s,\n true:%s' % (class_names[y_pred[idx]], 
                                        class_names[y_true[idx]]), size=8)
        if count > 16 : break
plt.tight_layout()
plt.show()
        



# 6) model 배포

# model.save('model/fashionMnist.keras')



# %%

from tensorflow.keras.models  import load_model
load_model = load_model('model/fashionMnist.keras')


# pip install opencv-python
import cv2

class_names=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



def prediction(filename):
    img=cv2.imread(filename)
    x=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x=x.reshape(1, 28*28)/255
    results = load_model.predict(x)
    result = np.argmax(results, axis=1) # (1,1)
    print(result)
    return result[0]

y=prediction('image/fashion_20.jpg')
    
print(class_names[y])























