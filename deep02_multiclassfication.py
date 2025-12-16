# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 09:58:44 2025

@author: letuin
"""
# %%
import tensorflow as tf
from sklearn.datasets import load_iris
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn  as sns

# %%
###  1 데이터 수집

iris=load_iris()
X = iris.data
y=iris.target
np.unique(y)  # multi classfication

# 인덱스 섞기
# idx=np.random.permutation(len(X))
# X=X[idx]
# y=y[idx]

# In['preprocess']
### 2. 데이터 전처리
# 2-1 X 데이터 정규화
tX = (X-X.min())/(X.max()-X.min())

# 2-2 데이터 분리 (학습, 테스트) 8:2

# train_idx = int(len(tX)*0.8) 
# train_x = tX[:train_idx]
# Ttrain_y = y[:train_idx]


# test_x = tX[train_idx:]
# Ttest_y = y[train_idx:]

train_x, test_x, Ttrain_y, Ttest_y = \
             train_test_split(tX, y, test_size=0.2, random_state=42)  #seed

# 2-3 OneHot encoding

train_y=tf.keras.utils.to_categorical(Ttrain_y, num_classes=3)
test_y=tf.keras.utils.to_categorical(Ttest_y, num_classes=3)


# In['model setting']
#  model setting
model=Sequential([
    Dense(units=10, activation='relu' , input_shape=(4,)),
    Dense(units=8, activation='relu' ),  
    Dense(units=3,  activation='softmax' )   ]) # 다중 분류

model.summary()

model.compile(optimizer="adam",
              loss="categorical_crossentropy", # 다중 분류
              metrics=['accuracy'])


history= model.fit(train_x, train_y, 
                   epochs=50, 
                   batch_size=8, 
                   validation_split=0.2)


# 시각화
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], 'b-', label='accuracy')
plt.plot(history.history['val_accuracy'], 'r--', label='val_accuracy')
plt.xlabel('Epoch')
plt.legend()


#  0.576923	0.448718	0.115385	0.0128205


# In['evaluation']
model.evaluate(test_x, test_y)
result = model.predict(test_x)

test_y[0]   # array([0., 0., 1.])
result[0]   # array([0.01754737, 0.43364468, 0.5488079 ],
test_y[12]  # 
result[12]  # 
np.argmax(result[0:10], axis=-1)
np.argmax(test_y[0:10], axis=-1)
cm=confusion_matrix(np.argmax(test_y, axis=-1),
                    np.argmax(result, axis=-1))
plt.figure(figsize=(7,7))
sns.heatmap(cm, annot= True, fmt='d' ,cmap='Blues')
plt.xlabel('predicted label', fontsize = 15)
plt.ylabel('true label', fontsize = 15)
plt.xticks(range(3),[0,1,2],rotation=45)
plt.yticks(range(3),[0,1,2],rotation=0)
plt.show()












