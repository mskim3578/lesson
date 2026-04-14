import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'   # Windows
# 마이너스 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib  
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score




# %%

def load_csv(CSV_PATH="data/semi_process_data2.csv"):
    
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    
    # 결측치 / 단일값 컬럼 삭제
    df = df.dropna(axis=1)
    df = df.drop(columns=[c for c in df.columns if df[c].nunique() <= 1])
    df['Depo_THK']=df['POST THK AVG']-df['PRE THK AVG']
    df['Particle']=df['POST PC']-df['PRE PC']
    
    
    def labeling_pro(target):
        mean_val=df[target].mean()
        std_val=df[target].std()
        
        lower_limit = mean_val - (2*std_val)
        upper_limit = mean_val + (2*std_val)
        
        y=np.where((df[target] < lower_limit) | 
                               (df[target] > upper_limit), 1, 0)
        
        return y
    
    df['Depo_THK_y']=labeling_pro('Depo_THK')
    df['Particle_y']=labeling_pro('Particle')
    
    
    
    return df




# %%
def hist_pro(df):
    
    fig=plt.figure(figsize=(10, 6))  
    count=1
    for col in df.columns:
        plt.subplot(4,7, count)
        count +=1
        plt.title(col, size=8)
        plt.hist(df[col])
    plt.tight_layout()
    return fig
    
def plot_pro(df):
    
    fig=plt.figure(figsize=(10, 6))  
    count=1
    for col in df.columns:
        plt.subplot(4,7, count)
        count +=1
        plt.title(col, size=8)
        plt.plot(range(len(df)),df[col])
    plt.tight_layout()
    return fig 

    
def corr_pro(df, target, threshold):    
    df = df.select_dtypes(include='number')  
    corr_matrix = df.corr()[target].abs()
   
    fig, ax = plt.subplots(figsize=(10, 6)) 
    
    filtered_corr = corr_matrix[corr_matrix > threshold]
    sorted_cols = filtered_corr.sort_values(ascending=False).index
  
    sns.heatmap(df[sorted_cols].corr(), 
                annot=True,           # 숫자 표시
                fmt=".2f",            # 소수점 둘째자리까지
                cmap='YlGnBu',        # 색상 테마 (Yellow-Green-Blue 혼합으로 부드러운 전개를 보여줌.)
                ax=ax, # 특정 서브플롯 위치 지정 
                # mask=mask,            # 마스크 적용
                linewidths=0.5,       # 칸 사이 구분선
                cbar_kws={"shrink": .8}) # 컬러바 크기 조절
   
    plt.title(f"공정 변수 간 상관관계 히트맵 (정렬 기준: {target})",
              fontsize=15, pad=20)
    return fig 
    
    
# %%    

def binary_model(df, target):
    
    models = {
    # 1. 선형 및 확률 기반 모델
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    
    # 2. 거리 및 커널 기반 모델
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(),
    
    # 3. 앙상블 (배깅)
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    
    # 4. 앙상블 (부스팅) - 실무에서 가장 많이 쓰임
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM":LGBMClassifier(
    random_state=42,
    verbosity=-1,               # 경고 메시지 출력 안 함
    n_estimators=100,           # 반복 횟수 (데이터가 적다면 조정)
    learning_rate=0.05,         # 학습률 (조금 낮춰서 세밀하게 학습)
    num_leaves=31,              # 트리 복잡도 (데이터가 적으면 15~20으로 낮추기)
    min_child_samples=10,       # 리프 노드에 필요한 최소 데이터 (경고 방지용으로 조정)
    importance_type='gain'      # 피처 중요도를 계산할 때 '이득(gain)' 기준 사용
        )
    
    }
    # 1. load_csv
    # 2. 전처리 
    # 뒤에 4개의 컬럼을 지운다  X
    feature_cols = df[df.columns[:-4]].select_dtypes(include='number').columns.tolist()
    
    # X 정규화 
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    
    # y
    y=df[target+"_y"]
    anormal = np.sum(y == 1)
    # 5. 결과 확인
    print(feature_cols)
    print("\n--- 라벨링 결과 요약 ---")
    print(anormal, (anormal/len(df)*100), '%') # 비율 확인 (약 4.5%가 1로 나와야 함)
    
    # 데이터 train , test 분리
    X_train, X_test, y_train, y_test = \
        train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    results = []
    # 3 model 작업
    
    fig = plt.figure(figsize=(10,30))  # (width, height)
    count=1
    
    for name,  model in models.items():
        
        # 학습
        model.fit(X_train, y_train)        
        # 예측
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred)    
        # 결과 저장
        results.append({
           "Model명": name,
           "model" : model,
           "scaler" : scaler,
           "Accuracy": acc,  #전체 데이터 중 맞게 맞춘 비율이 얼마인가
           "Precision": pre, #모델이 '불량'이라고 한 것들 중, 진짜 '불량'은 몇 개인가?
           "Recall": rec,    #실제 '불량'인 것들 중, 모델이 찾아낸 '불량'은 몇 개인가
           "F1-Score": f1  #정밀도(Precision)와 재현율(Recall)의 조화로운 평균값
        })
        
        plt.subplot(6, 2, count)
        cm = confusion_matrix(y_test, y_pred) 
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal (0)', 'Fault (1)'], 
                    yticklabels=['Normal (0)', 'Fault (1)'])
        plt.title('Confusion Matrix:'+name)
        plt.xlabel('Predicted')
        plt.ylabel('Actual Label')
        
        
        plt.subplot(6, 2, count+1) 
        plt.hist(x=y_test, alpha=0.5, color='b', 
                 label=str(np.sum(y_test == 0))+"/"+str(np.sum(y_test == 1)))
        plt.hist(x=y_pred+0.25, alpha=0.5, color='r', 
                 label=str(np.sum(y_pred == 0))+"/"+str(np.sum(y_pred == 1)))
        plt.xlabel('Actual Values (y_test)')
        plt.ylabel('Predicted Values (y_pred)')
        plt.title(name)
        plt.legend()
        plt.grid(True)
        count +=2    
        
        
        
        
     
    joblib.dump(results, 'model/binarymodel_results.pkl')

    # 나중에 다시 불러올 때
    # loaded_results = joblib.load('model/binarymodel_results.pkl')    
        
    df_results = pd.DataFrame(results).sort_values(by='F1-Score', ascending=False)
    print(df_results[['Model명','Accuracy','Precision','Recall','F1-Score']])
    return df_results[['Model명','Accuracy','Precision','Recall','F1-Score']], fig


# %%

def linear_model(df, target):
    # 1. 회귀 모델 딕셔너리 생성   
    
    models = {
    "Linear Regression": LinearRegression(),
    "Ridge (L2)": Ridge(alpha=1.0),
    "Lasso (L1)": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=100, random_state=42)
    }
    
    y=df[target]  #Linear 
    feature_cols = df[df.columns[:-4]].select_dtypes(include='number').columns.tolist()
    
    # X 정규화 
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    
    X_train, X_test, y_train, y_test = \
        train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    results = []
    # 3 model 작업
    
    fig = plt.figure(figsize=(10,30))  # (width, height)
    count=1
    
    # 3. 루프를 이용한 학습 및 예측
    for name, model in models.items():
            # 학습
          model.fit(X_train, y_train)      
          # 예측
          y_pred = model.predict(X_test)      
          # 성능 지표 계산
          mae = mean_absolute_error(y_test, y_pred)
          mse = mean_squared_error(y_test, y_pred)
          rmse = np.sqrt(mse)
          r2 = r2_score(y_test, y_pred)      
          # 결과 저장
          results.append({
              "Model명": name,
              "model" : model,
              "scaler" : scaler,
              "MAE": mae,   #MAE (Mean Absolute Error, 평균 절대 오차)
              "RMSE": rmse, #RMSE (Root Mean Squared Error, 평균 제곱근 오차)
              "R2 Score": r2 #R2 Score (R-Squared, 결정 계수)
          })
          plt.subplot(7, 1, count)
          # 산점도 그리기
          plt.scatter(x=y_test, y=y_pred, alpha=0.5)
            
          # 기준선 (y=x) 그리기
          line_coords = [y_test.min(), y_test.max()]
          plt.plot(line_coords, line_coords, color='red', 
                   linestyle='--', lw=2, label='Perfect Prediction')
            
          plt.xlabel('Actual Values (y_test)')
          plt.ylabel('Predicted Values (y_pred)')
          plt.title(f"{target}:{name}[{r2:.6f}]")
          plt.legend()
          plt.grid(True)               
          count +=1
          
    plt.tight_layout() 
      # 4. 결과 테이블 생성 및 출력 (R2 Score가 높을수록 좋은 모델)
    
    joblib.dump(results, 'model/linearmodel_results.pkl')
    
    df_results = pd.DataFrame(results).sort_values(by='R2 Score', 
                                                   ascending=False)
    # print(results)
    
    return df_results[['Model명','MAE','RMSE','R2 Score']], fig
     
      
    
    
    


# %%

def binary_predict(df, target):
    results = joblib.load('model/binarymodel_results.pkl') 
    name=results[0]['Model명']
    model=results[0]['model']
    scaler=results[0]['scaler']
    
    y=df[target+"_y"]
    
    
    feature_cols = df[df.columns[:-4]].select_dtypes(include='number').columns.tolist()
    
    # X 정규화    
    X_scaled = scaler.transform(df[feature_cols])
    
    y_pred = model.predict(X_scaled)
    df_anormal = df[df[target+"_y"] != y_pred]
    
    cm=confusion_matrix(y, y_pred) # 혼동 행렬
    print(cm) 
    
    tn, fp, fn, tp = cm.ravel()
  
      # 혼동 행렬의 각 요소를 다음과 같이 정의할 때:
      # TP (True Positive): 양성을 양성으로 맞게 예측
      # TN (True Negative): 음성을 음성을 맞게 예측
      # FP (False Positive): 음성을 양성으로 틀리게 예측 (1종 오류)
      # FN (False Negative): 양성을 음성으로 틀리게 예측 (2종 오류)         
      # 공식 적용
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
      
    print(f'{name}')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")    #F1-Score는 정밀도와 재현율을 결합한 지표입니다.
      
    
    
    df_cm = pd.DataFrame(cm)
    return df_cm, df_anormal
# %%    
# df_ai=load_csv()    
# df_results=binary_model(df_ai, 'Depo_THK')    
# df_results.iloc[:, []]














