# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 10:59:10 2026

@author: Admin
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC   # model.feature_importances_  없음 
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


import matplotlib.pyplot as plt
import seaborn as sns
import joblib


#  pip install scikit-learn
#  pip install xgboost lightgbm



# %%
def load_csv(CSV_PATH="data/semi_process_data2.csv"):           
    
    df = pd.read_csv(CSV_PATH, encoding="utf-8")        
    #1. 결측치 / 단일값 컬럼 삭제
    df = df.dropna(axis=1)
    df = df.drop(columns=[c for c in df.columns if df[c].nunique() <= 1])       
    
    #2. Depo_THK, Particle  추가    
    df['Depo_THK']=df['POST THK AVG']-df['PRE THK AVG']  
    df['Particle']=df['POST PC']-df['PRE PC']
    df['Particle'] = df['Particle'].clip(lower=0)    # lower=0 옵션을 주면 0을 최하한선으로 고정하겠다는 의미입니다.
    # np.log1p(값은 0보다 커야 한다 )    
    ### 3*시그마로 유/불 ??????
    def labeling_pro(target):
       mean_val = df[target].mean()
       std_val = df[target].std()    
       # 3. 2시그마 경계값 설정 (상한선, 하한선)
       lower_limit = mean_val - 3 * std_val
       upper_limit = mean_val + 3 * std_val
       
       y = np.where((df[target] < lower_limit) | 
                              (df[target] > upper_limit), 1, 0)
       #우량=0(Negative), 불량=1(Positive)
       return y
   
    df['Depo_THK_y']=labeling_pro('Depo_THK')
    df['Particle_y']=labeling_pro('Particle')
    
    return df

'''
df_all=load_csv() 


'''  

# %%  
def binary_model(df, target):   
    models = {
     # 1. 선형 및 확률 기반 모델
     "LogisticRegression": LogisticRegression(max_iter=1000, # 반복횟수
                                              C=0.5,  # 1 ~ 0 작을 수록 규제가 강해진다  
                                              class_weight={0: 1, 1: 32}, # 0: 우량(97%), 1(3%): 불량 비율 
                                              random_state=42),   # seed value
     
     # 2. 거리 및 커널 기반 모델
     # "SVM": SVC(probability=True, class_weight={0: 1, 1: 32}, random_state=42),
     "KNN": KNeighborsClassifier(),
     
     # 3. 앙상블 (배깅) : 
     # 복원 추출을 통해 여러 개의 서로 다른 샘플 데이터를 만들고(Bootstrap), 각각 학습시킨 뒤 결과를 하나로 합치는(Aggregating) 방법
     "RandomForestClassifier": RandomForestClassifier(n_estimators=100,    # 알상불 내부에 만들 결정 트리 개수 임
                                                      class_weight={0: 1, 1: 32},  # 1: 우량(97%), 0(3%): 불량 비율 
                                                      random_state=42),
     
     # 4. 앙상블 (부스팅) - 실무에서 가장 많이 쓰임
     # 이전 나무가 틀린 오차를 다음 나무가 순차적으로 보완해 나가는 '부스팅(Boosting)
     "XGBoost": XGBClassifier(
         scale_pos_weight=32,    # 0: 우량(97%), 1(3%): 불량 비율 
         eval_metric='logloss', 
         # logloss(교차 엔트로피 손실)는 모델이 예측한 확률이 실제 정답과 얼마나 다른지 패널티를 부여하며 계산합니다.
         # 0과 1을 분류하는 이진 분류(Binary Classification) 문제에서 가장 표준적이고 안정적인 평가지표
         random_state=42),
     
     
     "LightGBM":LGBMClassifier( 
         class_weight={0: 1, 1: 32},   # 1: 우량(97%), 0(3%): 불량 비율 
         random_state=42,
         verbosity=-1,               # 경고 메시지 출력 안 함
         n_estimators=100,           # 반복 횟수 (데이터가 적다면 조정) 오차를 보완해 나갈 순차적인 트리(부스팅 단계)의 개수
         learning_rate=0.05,         # 학습률 (조금 낮춰서 세밀하게 학습)
         num_leaves=15,              # 트리 복잡도 (데이터가 적으면 15~20으로 낮추기)
         min_child_samples=10,       # 리프 노드에 필요한 최소 데이터 (경고 방지용으로 조정)         
         importance_type='gain'      # 피처 중요도를 계산할 때 '이득(gain)' 기준 사용
         )
         # '''
         # 전체 데이터 수가 매우 많아서 3%에 해당하는 불량 데이터의 절대적인 개수가 수천 건 이상으로 충분하다면,
         # num_leaves=31,
         # min_child_samples=20,   # 데이터가 많으므로 최소 샘플 수도 늘려줌
         # '''
         }
        
    y=df[target+"_y"]    
    anormal = np.sum(y == 1)  # 불량
    # 5. 결과 확인
    print("\n--- 라벨링 결과 요약 ---")
    print(anormal, (anormal/len(df)*100), '%') # 비율 확인 (약 3.3%가 1로 나와야 함)
    
        
    # 꼭 지운다 !!!!!!
    ''' 
    ---  지워야 하는 feature
    Depo_THK: POST THK AVG, POST PC, Particle
    Particle: POST PC, POST THK AVG, Depo_THK
    
    ----- 살려두고 학습에 사용해야 하는 컬럼 (Features):
    PRE PC (공정 전 데이터이므로 사용 가능)
    PRE THK AVG (공정 전 데이터이므로 사용 가능)
    RF Power, Precursor flow rate, Chamber pressure 등의 순수 공정 센서 데이터
    ''' 
    
    
    cols= df.columns[:-4].tolist() 
    cols.remove('POST THK AVG')
    cols.remove('POST PC')  
    
    feature_cols = df[cols].select_dtypes(include='number').columns.tolist()
    print(feature_cols)
    
    #############
    # 1. 데이터 분할 (분류 모델이므로 stratify=y(유,불을 동일한 비율로)  권장)
    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_cols], y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. 스케일러 선언 및 적용 정규화
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Train 데이터로 fit & transform
    X_test_scaled = scaler.transform(X_test)        # Test 데이터는 transform만
    
    
    # 모델 평가 직전이나 직후에 아래 코드를 넣어 화면이나 터미널에 찍어보세요.
    print("실제 Test 데이터의 정상/불량 개수:", pd.Series(y_test).value_counts().to_dict())
########################    
    results=[]
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)     # 학습  
        y_pred = model.predict(X_test_scaled)  # 예측
        acc = accuracy_score(y_test, y_pred)     # 성능 지표 계산
        pre = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred)        
        # 결과 저장
        results.append({
            "Model명": name,            "model" : model,
            "scaler" : scaler,            "Accuracy": acc,  
            "Precision": pre, 
            "Recall": rec,    
            "F1-Score": f1 , #정밀도(Precision)와 재현율(Recall)의 조화로운 평균값
            "y_pred" :  y_pred,
            "feature_cols" : feature_cols  })
        print(name, '예측불량', np.sum(y_pred == 1), '실제 분량',np.sum(y_test == 1) )   # 예측 불량
        # 3. 결과 테이블 생성 및 출력
        #F1-Score는 정밀도와 재현율을 결합한 지표입니다.
        #F1-Score는 1(또는 100%)에 가까울수록 모델의 성능이 좋다고 판단
    df_results = pd.DataFrame(results).sort_values(by='F1-Score', ascending=False)
    df_results = df_results.reset_index(drop=True)
    print(df_results)
    '''    
    import matplotlib.pyplot as plt
    import seaborn as sns
    '''
    
    fig = plt.figure(figsize=(10,30))  # (width, height)
    count=1
    for row in range(len(df_results)) :    
        t_pred=df_results.loc[row]['y_pred']
        t_name=df_results.loc[row]['Model명']
   
        plt.subplot(6, 2, count)
        cm = confusion_matrix(y_test, t_pred) 
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Fault (0)', 'Normal (1)'], 
                    yticklabels=['Fault (0)', 'Normal (1)'])
        plt.title('Confusion Matrix:'+t_name)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')      
        
        plt.subplot(6, 2, count+1) 
        plt.hist(x=y_test, alpha=0.5, color='b', label="실제"+str(np.sum(y_test == 0))+"/"+str(np.sum(y_test == 1)))
        plt.hist(x=y_pred+0.25, alpha=0.5, color='r', label="예측"+str(np.sum(t_pred == 0))+"/"+str(np.sum(t_pred == 1)))
        plt.xlabel('Actual Values (y_test)')
        plt.ylabel('Predicted Values (y_pred)')
        plt.title(t_name)
        plt.legend()
        plt.grid(True)        
        count +=2 
    plt.tight_layout()
    plt.show()
    
    # 파일로 저장   
    setmodel=df_results.loc[0]
    t_name=setmodel['Model명']
    print(t_name)
    # import joblib
    joblib.dump(setmodel, f'model/binary_{target}.pkl')
    # 확인: 첫 번째 모델로 예측해보기
    #first_model = loaded_results[0]['model']
    # prediction = first_model.predict(X_test)
    return fig, df_results[['Model명','Accuracy','Precision','Recall','F1-Score']]

'''
df_all.columns
binary_model(df_all, 'Depo_THK')
binary_model(df_all, 'Particle')
'''    
def binary_predict(df, target):
     results = joblib.load(f'model/binary_{target}.pkl')  #가장 효율성이 좋은 모델을 저장후 사용 
     model=results['model']  
     scaler=results['scaler']   
     name=results['Model명']
     feature_cols=results["feature_cols"]   #model 생성시 기준 columns
     # 1. 피처 중요도 추출 및 데이터프레임 변환
     importance_df = pd.DataFrame({
         'Feature': feature_cols,
         'Importance': model.feature_importances_
     })     
     # 2. 중요도 피처 내림차순 정렬
     importance_df = importance_df\
            .sort_values(by='Importance', ascending=False).reset_index(drop=True)   
     print(importance_df)
     
     y=df[target+"_y"]  
     X_scaled=scaler.transform(df[feature_cols])      
     
     y_pred=model.predict(X_scaled)
     
     df_anormal = df[(y_pred == 1)]  # 0: 우량, 1: 불량
     
     
     print(df_anormal.head())
     
     cm=confusion_matrix(y, y_pred) # 혼동 행렬
     print(cm, "===========") 
     # return name, cm, df_anormal, importance_df

'''
binary_predict(df_all, 'Depo_THK')
'''    
    
    
    
    
    
    
    
    
    

