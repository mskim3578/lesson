# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 12:27:32 2026

@author: letuin
"""

import numpy as np
import pandas as pd
# 실습을 위한 가상의 데이터프레임 생성
data = {
"A": [1, 2, np.nan, 4, 4],
"B": [5, np.nan, np.nan, 8, 8],
"C": ["정상", "오류문자", "정상", "정상", "정상"],
}
df = pd.DataFrame(data)
# 1. df.isnull() 결측값 처리
print(df.isnull())
# - notnull() : 결측값인 경우 False, 일반값인 경우 True
print("\n[notnull() 결과]")
print(df.notnull())


# - dropna() : 결측값 제거 함수
# dropna(axis=1, thresh=2) : 정상값이 thresh=2개 미만인 컬럼 제거
print(df.dropna(axis=1, thresh=2))
# 02. 데이터 전처

# dropna(subset=['컬럼명'], how='any', axis=0) : 특정 컬럼에 결측값이 한 개라도 있는 행 제

print(df.dropna(subset=["A"], how="any", axis=0))
# - fillna() : 결측값 치환
# fillna(치환값, inplace=True) -> 여기서는 복사본으로 확인
print(df.fillna(0))
# fillna(method='ffill' 또는 'bfill') : 앞의 값(ffill) 또는 뒤의 값(bfill)으로 치환
# ※ 최신 pandas 버전에서는 method 대신 ffill() 또는 bfill() 함수 사용을 권장합니다.
print(df.ffill())
      
      
# 2. 중복데이터 처리
data = {
    '과일': ['사과', '바나나', '사과', '포도', '바나나'],
    '수량': [10, 20, 10, 15, 20],
    '지역': ['서울', '부산', '서울', '대구', '부산'],
    "오류": ["정상", "오류문자", "정상", "정상", "정상"],
}
df = pd.DataFrame(data)
# - duplicated() : 중복데이터 찾기. 첫번째 데이터는 False, 같은 데이터인 경우 두번째
True
print(df.duplicated())
# - drop_duplicates() : 중복데이터를 제거. 중복된 데이터 중 한개는 남김.
print(df.drop_duplicates())
# 3. 오류데이터 존재
# - 결측값 치환 : 결측값(np.nan)으로 만들고 처리하기
# replace(오류문자열, np.nan, inplace=True)

df_replaced = df.copy() # 원본 보존을 위해 카피본 사용
df_replaced.replace("오류문자", np.nan, inplace=True)
print(df_replaced)





import pandas as pd
import numpy as np
# --- [준비 단계] 소스 코드 테스트를 위한 샘플 데이터 생성--
data = {
'WaferNo': ['W_01', 'W_02', 'W_03', 'W_01', 'W_02'],
'Date_str': ['20260630_A', '20260630_B', '20260701_A', '20260701_B', '20260702_A'],
'Equipment': ['Eq-01', 'Eq-02', 'Eq-01', 'Eq-03', 'Eq-02'],
'DateTime': ['2026-06-30 20:15:00', '2026-06-30 21:30:00', '2026-07-01 08:45:00', 
'2026-07-01 20:00:00', '2026-07-02 11:10:00'],
'Raw_ID': ['20260630_LotA_01', '20260630_LotB_02', '20260701_LotA_03', 
'20260701_LotB_04', '20260702_LotA_05'],
'Temp': [180.5, 181.2, 179.8, 182.0, 180.1] # 가스 유량이나 온도를 대변하는 샘플 컬럼
}
df = pd.DataFrame(data)
df.info()









df['WaferNo_split'] = df['WaferNo'].str.split('_') # 분리 결과 확인용
df['Date_str_4'] = df['Date_str'].str[:4]
df['Equipment'] = df['Equipment'].str.replace('Eq-', '')
# #날짜형 타입 처리
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['Year'] = df['DateTime'].dt.year
df['Month'] = df['DateTime'].dt.month
df['Day'] = df['DateTime'].dt.day
df['Hour'] = df['DateTime'].dt.hour
df['DayName'] = df['DateTime'].dt.day_name()


# # 1. 문자열 split으로 날짜/시간 부분 분리하기
df['Date_part'] = df['Raw_ID'].str.split('_').str[0]
# # 2. 날짜 타입으로 제대로 바꾸기
df['Date_clean'] = pd.to_datetime(df['Date_part'], format='%Y%m%d')
df.info()
# # 3. 날짜별로 묶어서 가스 유량이나 온도 평균 확인하기
grouped_result = df.groupby('Date_clean')['Temp'].mean()



import pandas as pd
import glob

# 1. 특정 폴더 안의 모든 CSV 파일 경로 가져오기 (예: 'data_folder' 안의 csv 파일들)
# 현재 경로라면 '*.csv' 라고 적으시면 됩니다.
file_paths = glob.glob('data_folder/*.csv') 
# 2. 리스트 컴프리헨션으로 모든 파일을 읽어서 리스트에 담기
df_list = [pd.read_csv(file) for file in file_paths]
# 3. 위아래로 한 번에 병합하기 (행 방향으로 쌓기)
# ignore_index=True를 주면 기존 파일들의 인덱스를 무시하고 0부터 새로 번호를 매깁니다.
df_total = pd.concat(df_list, axis=0, ignore_index=True)
print(f"합쳐진 파일 개수: {len(file_paths)}개")
print(f"최종 데이터 형태: {df_total.shape}")



# --- [1. 두 개의 서로 다른 파일 읽기 (올려주신 소스)] ---
folder='data_folder'
df_fdc = pd.read_csv(f'{folder}/fdc_sensor_data.csv')
df_et = pd.read_csv(f'{folder}/et_quality_data.csv')

df_inner = pd.merge(df_fdc, df_et, on='WaferNo', how='inner')
# ② Outer Join (합집합: 한쪽에만 있어도 모두 포함, 없는 값은 NaN)
df_outer = pd.merge(df_fdc, df_et, on='WaferNo', how='outer')
df_left = pd.merge(df_fdc, df_et, on='WaferNo', how='left')
df_right = pd.merge(df_fdc, df_et, on='WaferNo', how='right')



# pip install scipy
from scipy import stats

# 1. 가상 데이터 준비 (설비 A, B, C의 수율 데이터)
group_A = np.array([85, 88, 89, 86, 90])
group_B = np.array([92, 89, 94, 93, 91])
group_C = np.array([80, 83, 82, 85, 81])

# 2. 일원분산분석(One-way ANOVA) 실행
f_stat, p_value =stats.f_oneway(group_A, group_B, group_C)


# 3. 결과 출력
print("=== Scipy ANOVA 결과 ===")
print(f"F-통계량 (F-statistic) : {f_stat:.4f}")
print(f"p-값 (p-value)          : {p_value:.6f}")
# 4. 유의수준 0.05 기준으로 판정
alpha = 0.05
if p_value < alpha:
    print("=> 판정: p-value가 0.05보다 작으므로 귀무가설 기각. 세 그룹의 평균 수율은 통계적으로 유의미한 차이가 있다.")
else:
    print("=> 판정: p-value가 0.05보다 크므로 귀무가설 채택. 세 그룹의 평균 수율 차이가 있다고 볼 수 없다.")




############  ols
# pip install statsmodels
import seaborn as sns
import matplotlib.pyplot as plt

df_sample=pd.read_csv('data/lot_sample_1000.csv')
df_sample.columns




# 상관계수 heatmap
tcorr=df_sample[df_sample.columns[3:-1]].corr()
sns.heatmap(tcorr,annot=True, cmap='coolwarm')
plt.show()



import statsmodels.api as sm
# pip install statsmodels

x=df_sample[df_sample.columns[4:-1]]
x=sm.add_constant(x)
y1=df_sample['PECVD_Thick']


model=sm.OLS(y1,x).fit()

# 4. 분석 결과 요약 리포트 출력
print("==== [2] OLS 모델 분석 결과 리포트 ====")
print(model.summary())
