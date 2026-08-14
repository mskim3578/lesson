
import numpy as np
import matplotlib.pyplot as plt
# pip install matplotlib

# %%  경사 하강법, 역전파(backpropagation)
# 
# 1. 가상 데이터
np.random.seed(42)
X = 2*np.random.rand(100,1)
y = 4 + 3 * X + np.random.randn(100,1)



# y=b + w*X
# 파라미터 설정
learning_rate=0.1 # 학습률 (보폭)
n_iterations=100
m=len(X)

# 
w=np.random.randn(1,1)
b=np.random.randn(1,1)

# y_pred = X.dot(w)+b  #  3*X + b=y
# print(f'X : {X}')
# print(f'w : {w}')
# print(f'y_pred : {y_pred}')

# 경사 하강법 루프 시작 
loss_history=[]

plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', alpha=0.5, label='Data')


for iteration in range(n_iterations):
    # 예측값 계산 (Y = wX + b)
    y_pred = X.dot(w)+b  #  3*X + b=y
    # 2) 손실 계산 (MSE: 평균제곱오차)
    loss = np.mean((y_pred - y) ** 2)  
    #  (2*(y_pred))/m
    loss_history.append(loss)
    
    # 편미분 계산 (Gradient 구하기) 
    # 3) 역전파(Backpropagation)
    w_gradient = (2/m) * X.T.dot(y_pred - y)
    b_gradient = (2/m) * np.sum(y_pred - y)
    # print(f'x.T : {X.T}')
    # print(f'(y_pred - y)  : {y_pred - y}')
    # print('#'*100)
    # 4) 가중치 업데이트 (경사의 반대 방향으로 이동)
    w = w - learning_rate * w_gradient
    b = b - learning_rate * b_gradient
    
    # 초반, 중반, 종반의 회귀선 그리기 (시각화)
    if iteration in [0, 1, 5, 10, 20, 49]:
          X_new = np.array([[0], [2]])
          y_predict_new = X_new.dot(w) + b 
          style = "r--" if iteration < 10 else "g-"
          if iteration == 49: style = "r-"
          print(w, b)
          plt.plot(X_new, y_predict_new, style, alpha=0.7)
         
plt.title("Regression Line Update Process")
plt.xlabel("X (Process Parameter)")
plt.ylabel("Y (Yield)")
plt.grid(True) 
# 오른쪽 그래프: 반복에 따른 Loss(오차) 감소 추이
plt.subplot(1, 2, 2)
plt.plot(range(n_iterations), loss_history, 'b-')
plt.title("Loss Reduction (MSE)")
plt.xlabel("Iteration (Epoch)")
plt.ylabel("Loss")
plt.grid(True)

plt.tight_layout()

plt.show()



# %%  bachnormalization
import tensorflow as tf
from tensorflow.keras import layers, models

# Keras Sequential API 예시
model = models.Sequential([
    # 1. Conv Layer + BatchNorm
    layers.Conv2D(16, (3, 3), padding='same', input_shape=(32, 32, 3)),
    layers.BatchNormalization(), # Keras가 채널 차원을 자동 인식
    layers.Activation('relu'),
    
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    
    # 2. Dense Layer + BatchNorm
    layers.Dense(128),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    
    layers.Dense(10)
])

# 모델 요약 출력
model.summary()

# Dummy 입력 데이터로 실행
dummy_input = tf.random.normal([8, 32, 32, 3])
output = model(dummy_input)
print("\n출력 형태:", output.shape) # (8, 10)


# %%  dataset dataloader
import torch
# pip install torch
from torch.utils.data import Dataset, DataLoader

# ================= ============================================
# 1. 커스텀 Dataset 정의
# ==============================================================
class CustomDataset(Dataset):
    def __init__(self, x_data, y_data):
        """
        데이터 초기화 및 데이터셋에 필요한 전처리/변환 정의
        """
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.FloatTensor(y_data)

    def __len__(self):
        """
        전체 데이터셋의 총 샘플 개수를 반환
        """
        return len(self.x_data)

    def __getitem__(self, idx):
        """
        인덱스(idx)가 주어졌을 때, 해당 위치의 데이터 샘플 1개를 반환
        """
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x, y


# ================= ============================================
# 2. Dataset 및 DataLoader 객체 생성
# ==============================================================
# 예시 데이터 생성 (총 100개 데이터 샘플, 입력 피처 5개, 타겟 1개)
raw_x = [[i * 0.1 for i in range(5)] for i in range(100)]
raw_y = [[i % 2] for i in range(100)]

# Dataset 인스턴스화
dataset = CustomDataset(raw_x, raw_y)
len(dataset)
# DataLoader 인스턴스화
dataloader = DataLoader(
    dataset=dataset,     # 불러올 Dataset 객체
    batch_size=16,       # 미니배치 크기 (한 번에 불러올 샘플 개수)
    shuffle=True,        # 에폭(Epoch)마다 데이터 순서를 섞을지 여부
    num_workers=0,       # 데이터 로딩에 사용할 서브 프로세스 수 (윈도우 환경은 0 권장)
    drop_last=False      # 마지막 남는 자투리 배치를 버릴지 여부
)


# ================= ============================================
# 3. 실제 학습 루프(Training Loop)에서의 사용 예시
# ==============================================================
epochs = 2

for epoch in range(epochs):
    print(f"\n--- Epoch {epoch + 1} ---")
    
    # DataLoader를 반복문(iterable)에 넣으면 batch_size 단위로 데이터를 꺼내옵니다.
    for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1:2d} | Input Shape: {list(batch_x.shape)} | Target Shape: {list(batch_y.shape)}")
        
# %%   deep ensemble 

# 효율성이 좋은 몇개의 모델을 함께 정용하는 모델을 만든다

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 사용할 알고리즘들 로드
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier   
# VotingClassifier  ensemble
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# 1. 가상의 공정 불균형 데이터 생성 (정상 99% : 불량 1%)
X, y = make_classification(
    n_samples=5000, n_features=10, 
    weights=[0.99, 0.01], flip_y=0, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. 비교할 개별 모델 정의
models = {
    'LogisticRegression': LogisticRegression(),
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier(),
    'RandomForestClassifier': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(random_state=42, verbose=-1)
}

# 3. 개별 모델 학습 및 질문하신 형태의 성능 비교 테이블 생성
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # 지표 산출
    results.append({
        'Model명': name,
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'Recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
        'F1-Score': round(f1_score(y_test, y_pred, zero_division=0), 4)
    })

df_results = pd.DataFrame(results)
print("=== [1] 개별 머신러닝 모델 성능 비교 ===")
print(df_results.sort_values(by='F1-Score', ascending=False).to_string(index=False))
print("\n" + "="*50 + "\n")


# 4. 상위 성능 모델들을 하나로 묶는 '보팅 앙상블(Voting Ensemble)' 구현
# 이미지에서 우수했던 상위 3개 알고리즘 조합
ensemble_clf = VotingClassifier(
    estimators=[
        ('xgb', models['XGBoost']),
        ('rf', models['RandomForestClassifier']),
        ('lgb', models['LightGBM'])
    ],
    voting='soft' # 확률 기반 다수결 투표
)

# 앙상블 모델 학습 및 평가
ensemble_clf.fit(X_train, y_train)
y_ensemble_pred = ensemble_clf.predict(X_test)

print("=== [2] 상위 모델 3개를 결합한 앙상블 성능 ===")
print(f"Accuracy  : {accuracy_score(y_test, y_ensemble_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_ensemble_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_ensemble_pred):.4f}")
print(f"F1-Score  : {f1_score(y_test, y_ensemble_pred):.4f}")


# %%    vgg
# 2. 이미지 읽기
import cv2
img = cv2.imread("output/face_origin.jpg")

if img is None:
    print("이미지를 불러오지 못했습니다. 경로를 확인하세요.")
    exit()

print("지브리 애니메이션풍 변환 시작 (CPU 연산)...")

# 3. 양방향 필터(Bilateral Filter)로 형태의 선은 살리고 표면 질감은 뭉개기
# 이 과정이 애니메이션 셀 채색 느낌을 줍니다.
color = cv2.bilateralFilter(img, d=9, sigmaColor=40, sigmaSpace=40)  #### 30, 40, 50

# 4. 에지(외곽선) 추출하여 만화 같은 테두리 만들기
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_blur = cv2.medianBlur(gray, 5)
edges = cv2.adaptiveThreshold(
    gray_blur, 255, 
    cv2.ADAPTIVE_THRESH_MEAN_C, 
    cv2.THRESH_BINARY, 
    blockSize=9, C=2
)

# 5. 수채화풍 효과(Stylization) 적용 (지브리 특유의 배경 느낌 추가)
# sigma_s: 공간 반경(크기), sigma_r: 범위 반경(색상 유사도)
cartoon_space = cv2.stylization(color, sigma_s=40, sigma_r=0.1)

# 6. 외곽선 합치기
# 애니메이션처럼 두꺼운 외곽선을 원할 때 활성화 (선택 사항)
# ghibli_style = cv2.bitwise_and(cartoon_space, cartoon_space, mask=edges)

# 7. 결과 저장
output_name = "output/ghibli_opencv.jpg"
cv2.imwrite(output_name, cartoon_space)

print(f"변환 완료! '{output_name}' 파일이 저장되었습니다.")
