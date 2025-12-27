# -*- coding: utf-8 -*-
#  렛유인 project1 
import openpyxl 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor



plt.rcParams['axes.unicode_minus'] = False  #chart에 마이너스 프린트 
plt.rc("font",family="Malgun Gothic")   # 한글 프린트   

def set_loaddata(filename=''):
    global all_sheets,  fname, sheetname
    if not filename : filename = "data1/Raw data_ref_C_EX1.xlsx"
    all_sheets = pd.read_excel(filename, sheet_name=None, header=None)
    all_sheets.keys()  # dictionary
    len(all_sheets.keys())
    sheetname = list(all_sheets.keys())
    fname=filename


def pre_process(sheetindex):
    global number_df, df
    
    ### 1. 엑셀 파일에 필요한 sheet 선택한다 
    df = all_sheets[sheetname[sheetindex]]   # 0 -> Yield, 1 -> ET(DC)
    
    ### 2. df.columns  setting
    # temp1 = df.iloc[0].fillna(method="ffill").to_list()  # deprecate 됬음 
    # df.ffill() nan 일때 앞의 값으로 채운다 
    # df.bfill() nan 일때 뒤의 값으로 채운다 
    # df.fillna('') nan 일때 '' 로 채운다 
    
    # col1 에서 null일 때  앞의 값으로 채운다 
    col1 = df.iloc[0].ffill().fillna('').to_list()
    col2 = df.iloc[1].ffill().fillna('').to_list() 
 
    # c1, c2 중에 ''이 있으면 '_'을 넣지 않는다 
    df.columns=[c1 + c2  if c1=='' or c2 == '' else c1 + "_"+ c2 for c1, c2 in zip(col1, col2)]
    # print(sheetname[sheetindex], df.columns)
    
    ### 3. 처음에 두개의 columns으로 사용한 row를 삭제한다
    df=df.drop(index=[0,1]) # df.iloc[0], df.iloc[1]  삭제
    
    ### 4. 숫자만 가지고 있는 정보를 확인 한다 
    number_col=df.columns[3:].to_list()  #No.	LOTID	WFID columns에서 제거한다 
    number_df=df[number_col]
    
    number_df=number_df.convert_dtypes()	# 자료마다 type변경을 한다
    
    # 숫자 타입 컬럼만 선택하여 새로운 DataFrame 생성
    # include='number'는 정수(int), 실수(float) 등 모든 숫자 타입을 포함합니다.
    number_df = number_df.select_dtypes(include='number') # number type만 었는다
    
    number_df.info()   # null을 확인 한다 
# %%
# 1. set_loaddata  :  excel 파일 번호 선택
excel_num="5"   # data1/Raw data_ref_C_EX1.xlsx
sheetindex=1    # sheetname : ['Yield', 'ET(DC)', 'ADI_ACI CD', 'Thickness', 'FDC', 'Item 설명']

set_loaddata("data1/Raw data_ref_C_EX"+excel_num+".xlsx")  #한개의 excel file을 읽어서 sheet별로 작업한다
pre_process(sheetindex)  # 전처리 
print(sheetname)
print(number_df.columns)

# 2. 상삼각 행렬(Upper triangle)만 선택 (중복 제거)
# 행렬에서 대각선을 기준으로 **오른쪽 위(Upper)**에 있는 값들만 1(True)로 남기고, 나머지는 0(False)으로 만듭니다.
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# 3. 상관계수가 0.9 이상인 컬럼들 찾기
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]

print(f"삭제 권장 피쳐 (중복 정보): {to_drop}")

# 4. 데이터프레임에서 제거
X_filtered = tdf.drop(columns=to_drop)

#  annot = True, fmt = '.2f',
plt.figure(figsize = (7, 7))
sns.heatmap(X_filtered.corr().abs(), cmap = 'Blues')
plt.xlabel('predicted label', fontsize = 15)
plt.ylabel('true label', fontsize = 15)
plt.show()

    
