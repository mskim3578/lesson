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

