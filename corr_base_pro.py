import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import project2.waferfunc as wf
import math




# 2. 순수 수식으로 상관계수를 구하는 함수 정의
def calculate_pure_correlation(list_a, list_b):
    n = len(list_a)
    
    # 2-1. 평균(Mean) 계산
    mean_a = sum(list_a) / n
    mean_b = sum(list_b) / n
    
    # 2-2. 분자(공분산의 분자 부분) 및 분모(각 변수의 제곱합) 초기화
    covariance_numerator = 0.0
    variance_a_sum = 0.0
    variance_b_sum = 0.0
    
    # 2-3. 시그마(합산) 기호 계산 구현
    for i in range(n):
        diff_a = list_a[i] - mean_a
        diff_b = list_b[i] - mean_b
        
        covariance_numerator += diff_a * diff_b  # ∑(x - x_bar)(y - y_bar)
        variance_a_sum += diff_a ** 2             # ∑(x - x_bar)^2
        variance_b_sum += diff_b ** 2             # ∑(y - y_bar)^2
        
    # 2-4. 분모 계산 (루트 씌우기)
    denominator = math.sqrt(variance_a_sum * variance_b_sum)
    
    # 분모가 0인 경우(변동이 없는 상수 데이터인 경우) 예외 처리
    if denominator == 0:
        return 0.0
        
    # 2-5. 최종 상관계수 반환
    return covariance_numerator / denominator


_, _, df_recipe = wf.load_data()
# %%
# 1. 5개의 변수 데이터 생성 (X1, X2, X3, X4, Y) - 각 10개의 데이터





features = df_recipe.columns[1:7]
target = 'mean'




# 3. 각 독립변수와 Target(Y) 간의 상관계수 계산 및 출력
print("--- 순수 계산식으로 구한 Target(Y)과의 상관계수 ---")
for f in features:
    corr_val = calculate_pure_correlation(df_recipe[f].to_list(), df_recipe[target].to_list())
    print(f"{f}와 Y의 상관계수: {corr_val:.2f}")