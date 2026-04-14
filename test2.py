from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score







  models = {
    "Linear Regression": LinearRegression(),
    "Ridge (L2)": Ridge(alpha=1.0),
    "Lasso (L1)": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=100, random_state=42)
    }
    


















from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

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















