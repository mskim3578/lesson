# -*- coding: utf-8 -*-


# pip install tensorflow

from tensorflow.keras.datasets.mnist import load_data  #데이터 수집
from tensorflow.keras.utils import to_categorical
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
# pip install scikit-learn
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix
import seaborn  as sns
# pip install seaborn

# %%

def makeImg(idx):
    tempimg = x_test[idx]
    timg=Image.fromarray(tempimg)
    timg.save(f'image/num_{idx}.jpg', 'jpeg')


#%%
#1)  데이터 수집
(Tx_train, Ty_train), (x_test, y_test) = load_data() #number


# %%
makeImg(30)

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
    Dense(units=64, activation='relu' , 
              input_shape=(784,)),
    Dense(units=32, activation='relu' ),  
    Dense(units=10,  activation='softmax' )   ])   # 다중 분류
   

model.compile(optimizer="adam",
              loss="categorical_crossentropy", # 다중 분류
              metrics=['acc'])

history= model.fit(x_train, y_train, 
                   epochs=30, 
                   batch_size=127, 
                   validation_data=(x_val, y_val))


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

# 4. 평가 (evaluate)
model.evaluate(x_test,y_test) 



# 5. 예측 (predict)
results = model.predict(x_test)
np.argmax(results[:10],axis=-1)
np.argmax(y_test[:10],axis=-1)

#  array([7, 2, 1, 0, 4, 1, 4, 9, 5, 9])



#예측이 틀린 이미지 16개 프린트 
count=0
for idx in range(len(results)) : #0~15까지
    number_sol = np.argmax(results,axis=1)[idx]
    number_y = np.argmax(y_test,axis=1)[idx]    
    
    if number_y == number_sol :        
        plt.subplot(4, 4, count+1)  #4행4열
        plt.axis('off') 
        plt.imshow(x_test[idx].reshape(28, 28)) #2차원배열. 그래프
        plt.title('Pred:%d,lab:%d' % (number_sol,number_y),
                          fontsize=15)
        count +=1
        if count > 15 : break
plt.tight_layout()
plt.show()











