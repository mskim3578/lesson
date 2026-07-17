# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 10:48:01 2026

@author: user
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
def load_data():
    
    df=pd.read_csv("data/semi_process_data2.csv")   
    
    #1. 결측치 / 단일값 컬럼 삭제
    df=df.dropna(axis=1)
    droplist = [c for c in df.columns if df[c].nunique() <= 1]    
    print(droplist)
    df=df.drop(columns=droplist)
    
    #2. Depo_THK, Particle추가
    df['Depo_THK']=df['POST THK AVG']-df['PRE THK AVG']
    df['Particle']=df['POST PC']-df['PRE PC']
    
    
    #3. labeling 우량, 불량 분류
    def labeling_pro(target):
        mean_val = df[target].mean()
        std_val = df[target].std()
        
        # ucl, lcl 확인
        ucl = mean_val + (std_val * 3)
        lcl = mean_val - (std_val * 3)
        
        y= np.where((df[target]< lcl)|(df[target]>ucl) , 1, 0) #1: 불량, 0: 우량
        
        return y
        
    df['Depo_THK_y']=labeling_pro('Depo_THK')   
    df['Particle_y']=labeling_pro('Particle')   
    
    
    
    return df

'''
df=load_data()
df['Depo_THK']=df['POST THK AVG']-df['PRE THK AVG']
df['Particle']=df['POST PC']-df['PRE PC']
'''

# %%

def hist_pro(df):
    fig=plt.figure(figsize=(10,6))
    count=1 
    wid=6 
    heig=len(df.columns)//wid + 1
    for col in df.columns:
        plt.subplot(heig, wid, count)
        count +=1 
        plt.title(col, fontsize=8)
        plt.hist(df[col])
        
    plt.tight_layout()
    return fig


def line_pro(df):
    df=df.select_dtypes(include='number')
    fig=plt.figure(figsize=(8,5))
    count=1 
    wid=6 
    heig=len(df.columns)//wid + 1
    for col in df.columns:
        plt.subplot(heig, wid, count)
        count +=1 
        ucl=df[col].mean() + (df[col].std()*3)
        lcl=df[col].mean() - (df[col].std()*3)
        
        plt.title(col, fontsize=8)
        plt.plot(range(len(df[col])), df[col])  
        plt.axhline(y=ucl , color='r', linestyle='dashed')
        plt.axhline(y=lcl , color='r', linestyle='dashed')
        
             
    
    plt.tight_layout()
    return fig

'''
line_pro(df)
hist_pro(df[['SiO2 Under SiN depo THK',
       'SiO2 Under SiN CMP THK', 'etch RF reflect power']])



'''


# %%

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
 
 
def binary_model(df, target):
   models={
       # 1. 선형 및 확률 기반 모델
   "LogisticRegression": LogisticRegression(max_iter=1000, # 반복횟수
                                            C=0.1,  # 1 ~ 0 작을 수록 규제가 강해진다  
                                            class_weight={0: 1, 1: 32},  # 0: 우량, 1: 불량 비율 
                                            random_state=42),   # seed value
      # 2. 거리 및 커널 기반 모델
     "SVM": SVC(probability=True, random_state=42),
     "KNN": KNeighborsClassifier(),   
      # 3. 앙상블 (배깅) : 
      # 복원 추출을 통해 여러 개의 서로 다른 샘플 데이터를 만들고(Bootstrap), 각각 학습시킨 뒤 결과를 하나로 합치는(Aggregating) 방법
     "RandomForestClassifier": RandomForestClassifier(n_estimators=100,    # 알상불 내부에 만들 결정 트리 개수 임
                                                       class_weight={0: 1, 1: 32},
                                                       random_state=42),
      
       # 4. 앙상블 (부스팅) - 실무에서 가장 많이 쓰임
      # pip install XGBoost
      "XGBoost": XGBClassifier(
       scale_pos_weight=32,    # class_weight={0: 1, 1: 32}
       eval_metric='logloss', 
       # logloss(교차 엔트로피 손실)는 모델이 예측한 확률이 실제 정답과 얼마나 다른지 패널티를 부여하며 계산합니다.
       # 0과 1을 분류하는 이진 분류(Binary Classification) 문제에서 가장 표준적이고 안정적인 평가지표
       random_state=42),     
      
       # pip install LightGBM
       # from lightgbm import LGBMClassifier, LGBMRegressor
       "LightGBM":LGBMClassifier( 
         random_state=42,
         verbosity=-1,               # 경고 메시지 출력 안 함
         n_estimators=100,           # 반복 횟수 (데이터가 적다면 조정) 오차를 보완해 나갈 순차적인 트리(부스팅 단계)의 개수
         learning_rate=0.05,         # 학습률 (조금 낮춰서 세밀하게 학습)
         num_leaves=15,              # 트리 복잡도 (데이터가 적으면 15~20으로 낮추기)
         min_child_samples=10,       # 리프 노드에 필요한 최소 데이터 (경고 방지용으로 조정)         
         importance_type='gain'      # 피처 중요도를 계산할 때 '이득(gain)' 기준 사용
         )
       
       }
   
   y=df[target+"_y"]
   anormal = np.sum(y==1)
   print(anormal, (anormal/len(df)*100), '%')
      
   
   
'''
df=load_data()
df.columns
binary_model(df, 'Depo_THK')

'''











