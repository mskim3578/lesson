# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 10:59:46 2025

@author: letuin




"""



# %%

import pandas as pd
import tensorflow as tf   # pip install tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import seaborn  as sns
from sklearn.metrics import confusion_matrix



# %%

#### 1)   data 수집
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"

red = pd.read_csv(url+'winequality-red.csv', sep=';') #red 와인 정보
white = pd.read_csv(url+'winequality-white.csv', sep=';') #white 와인 

red.info() #1599
white.info()  #4898

'''

1 - fixed acidity : 주석산농도
2 - volatile acidity : 아세트산농도
3 - citric acid : 구연산농도
4 - residual sugar : 잔류당분농도
5 - chlorides : 염화나트륨농도
6 - free sulfur dioxide : 유리 아황산 농도
7 - total sulfur dioxide : 총 아황산 농도
8 - density : 밀도
9 - pH : ph
10 - sulphates : 황산칼륨 농도
11 - alcohol : 알코올 도수
12 - quality (score between 0 and 10) : 와인등급
'''

#지도학습 ---  binary classfication
### 2. 전처리

# label
red['type']=0   # label 
white['type']=1 

#  dataframe 합친다
wine=pd.concat([red, white])

# min, max 정규화
wine.min()
wine.max()
wine_norm = (wine-wine.min())/(wine.max()-wine.min())
wine_norm.head()


wine_shuffle=wine_norm.sample(frac=1)
wine_shuffle.head()

# numpy type으로 변경
wine_np=wine_shuffle.to_numpy()

# 학습, 테스트 데이터 분리 8:2
train_idx = int(len(wine_np)*0.8) #5197

train_x=wine_np[:train_idx, :-1]  # :10
train_y=wine_np[:train_idx, -1]


test_x=wine_np[train_idx:, :-1]  # :10
test_y=wine_np[train_idx:, -1]

# one-hot encoding

train_y=tf.keras.utils.to_categorical(train_y, num_classes=2)
test_y=tf.keras.utils.to_categorical(test_y, num_classes=2)


# In['model setting']
#  model setting
model=Sequential([
    Dense(units=24, activation='relu' , input_shape=(12,)),
    Dense(units=24, activation='relu' ),
    Dense(units=12, activation='relu'),
    Dense(units=2,  activation='sigmoid' )   ]) # 이중 분류

model.summary()

model.compile(optimizer="adam",
              loss="binary_crossentropy", # 이중 분류
              metrics=['accuracy'])


history= model.fit(train_x, train_y, 
                   epochs=25, 
                   batch_size=32, 
                   validation_split=0.25)

# In[평가]
import matplotlib.pyplot as plt
# 시각화
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], 'b-', label='loss')
plt.plot(history.history['val_accuracy'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()

# 평가 
model.evaluate(test_x, test_y)
#[0.02040867879986763, 0.9953846335411072]

# In[predict]
result = model.predict(test_x)


test_y[0]   # array([0., 1.])
result[0]   # array([0.00267219, 0.9984468 ]

test_y[12]  # array([1., 0.])
result[12]  # array([9.9999756e-01, 1.4566966e-05]

np.argmax(result[16:26], axis=-1)
np.argmax(test_y[16:26], axis=-1)



cm=confusion_matrix(np.argmax(test_y, axis=-1),
                    np.argmax(result, axis=-1))

plt.figure(figsize=(7,7))
sns.heatmap(cm, annot= True, fmt='d' ,cmap='Blues')
plt.xlabel('predicted label', fontsize = 15)
plt.ylabel('true label', fontsize = 15)
plt.xticks(range(2),['red','white'],rotation=45)
plt.yticks(range(2),['red','white'],rotation=0)
plt.show()




























