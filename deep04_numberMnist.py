# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 14:34:37 2025
0......................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................
@author: letuin
"""
# %%
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.utils import to_categorical
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix
import seaborn  as sns

# %%

def makeImg(idx):
    tempimg = x_test[idx]
    timg=Image.fromarray(tempimg)
    timg.save(f'image/num_{idx}.jpg', 'jpeg')




#%%
#1)  데이터 수집
(Tx_train, Ty_train), (x_test, y_test) =load_data() #number Mnist



# %%
makeImg(55)

random_idx = np.random.randint(60000, size=3)

for idx in random_idx:
    img = Tx_train[idx, :]
    label= Ty_train[idx]
    plt.figure()
    plt.imshow(img)
    plt.title('%d-th data , label id %d' % (idx, label), fontsize=15)
    
# %%
#2) 데이터 전처리 
#2-1) val 자료 분리
x_train, x_val, y_train, y_val = \
             train_test_split(Tx_train, Ty_train, test_size=0.2, 
                              random_state=777)   
#   x_train,  x_test, x_val 정규화
x_train=x_train.reshape(48000, 28*28)/255
x_val=x_val.reshape(12000, 28*28)/255
x_test=x_test.reshape(10000, 28*28)/255
             
             
# y_train,  y_test,  y_val  oneHot  
y_train=to_categorical(y_train)
y_val=to_categorical(y_val)
y_test=to_categorical(y_test)

    

# In['model setting']
#  model setting
model=Sequential([
    Dense(units=64, activation='relu' , input_shape=(784,)),
    Dense(units=32, activation='relu' ),  
    Dense(units=10,  activation='softmax' )   ]) # 다중 분류

model.summary()

model.compile(optimizer="adam",
              loss="categorical_crossentropy", # 다중 분류
              metrics=['accuracy'])


history= model.fit(x_train, y_train, 
                   epochs=30, 
                   batch_size=127, 
                   validation_data=(x_val, y_val))



#%%
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

#%%
# In['evaluation']

model.evaluate(x_test, y_test)
result = model.predict(x_test)

y_test[0]   # array([0., 0., 1.])
result[0]   # array([0.01754737, 0.43364468, 0.5488079 ],
y_test[12]  # 
result[12]  # 
np.argmax(result[0:10], axis=-1)
np.argmax(y_test[0:10], axis=-1)
cm=confusion_matrix(np.argmax(y_test, axis=-1),
                    np.argmax(result, axis=-1))

plt.figure(figsize=(7,7))
sns.heatmap(cm, annot= True, fmt='d' ,cmap='Blues')
plt.xlabel('predicted label', fontsize = 15)
plt.ylabel('true label', fontsize = 15)
x_pos = np.arange(10)+0.5
plt.xticks(ticks=x_pos, labels=[0,1,2,3,4,5,6,7,8,9], rotation=45)
plt.yticks(ticks=x_pos, labels=[0,1,2,3,4,5,6,7,8,9], rotation=0 )

plt.show()