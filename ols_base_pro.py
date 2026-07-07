import numpy as np
import project2.waferfunc as wf
import pandas as pd
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# 1. 가상 데이터 생성 (독립변수 6개)
# np.random.seed(42)
# n_samples = 100
# n_features = 6


_, _, df_recipe = wf.load_data()




 # X 데이터 정규화


# 100행 6열의 무작위 독립변수 X 생성
cols = df_recipe.columns[1:7]
X= pd.DataFrame(scaler.fit_transform(df_recipe[cols]), columns=cols)
X=X.to_numpy()  #numpy로 변경 해야 한다 

# 정답 y 생성 (노이즈 추가)
# y = X @ true_W + true_b + np.random.randn(n_samples, 1) * 0.1
y=df_recipe['mean'].to_numpy()   #numpy로 변경 해야 한다 

n_samples = len(y)   # 541
n_features = len(cols)   #6

y=y.reshape(-1, 1)  # 2차원 이여야한다 
y=y+np.random.randn(n_samples, 1) * 0.1




# 2. 하이퍼파라미터 설정
learning_rate = 0.1
epochs = 2000
# %%

# 3. 가중치(W)와 편향(b) 초기화
# W는 6행 1열의 제로 행렬로 시작합니다.
W = np.zeros((n_features, 1))  # 변수의 종류 6가지 
b = 0.0

# 4. 경사하강법 학습 시작
for epoch in range(epochs):
    # 행렬 곱(@)을 이용해 100개 데이터의 예측값을 한 번에 계산 (y_pred shape: 100x1)
    y_pred = X @ W + b
    
    # 오차 계산
    error = y_pred - y
    
    # 비용 함수(MSE)의 미분값(Gradient) 계산
    # 행렬 연산을 이용하면 6개 변수의 기울기가 한 번에 계산됩니다.
    dW = (2 / n_samples) * (X.T @ error)
    db = (2 / n_samples) * np.sum(error)
    
    # 가중치와 편향 업데이트
    W -= learning_rate * dW
    b -= learning_rate * db
    
    # 200번마다 진행 상황 출력
    if epoch % 200 == 0:
        mse = np.mean(error ** 2)
        print(f"Epoch {epoch:4d}: MSE = {mse:.4f}")

# 5. 최종 결과 확인
print("\n--- 학습 완료 후 최종 가중치 ---")

# len(X) 대신 변수 개수인 n_features (6) 만큼만 반복합니다.
for i in range(n_features):
    # 실제 컬럼명(cols[i])을 같이 출력해주면 어떤 변수의 가중치인지 알기 쉽습니다.
    print(f"w{i+1} ({cols[i]}의 가중치): {W[i][0]:.4f}")
   

print(f"b   (Y 절편 / 편향)         : {b:.4f}")