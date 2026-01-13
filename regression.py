import openpyxl 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import warnings

# 모든 FutureWarnings 무시
# warnings.simplefilter(action='ignore', category=FutureWarning)  
warnings.simplefilter(action='ignore', category=FutureWarning)



# linear 알고리즘  임포트
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
# pip install xgboost lightgbm 


# %%  렛유인 project1 function

plt.rcParams['axes.unicode_minus'] = False  #chart에 마이너스 프린트 
plt.rc("font",family="Malgun Gothic")   # 한글 프린트   

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "DecisionTree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

#1.  데이터 수집, 한개의 df로 저장후 커럼 이름 정리후에 number 타입인 피쳐만 사용 한다 
def set_loaddata():
    global df_all_t, all_df, df_t, sheetList, col1, col2
    
    # sheet_name따라 column name이 달라서 5종류의 df를 만든다
    df_all_t=pd.DataFrame([]) 
    
    for i in range(1,6):  # excel file 번호   
      all_sheets = pd.read_excel(f"data1/Raw data_ref_C_EX{i}.xlsx", 
                                      sheet_name=None, header=None)
      df_t=pd.DataFrame([])
      #  전체 list columns으로 합친다       
      sheetList= list(all_sheets.keys())[:5] # 'Item 설명' 제외
      for key in sheetList:           
          one_sheet=all_sheets[key]  # data
          one_sheet=one_sheet.drop(columns=[0,1,2])          
          df_t=pd.concat([df_t, one_sheet], axis=1)   # column 추가
          df_t.columns=[i for i in range(len(df_t.columns))]
      
        
      # 5개의 excel 파일을 한개의 df_all로 완성
      df_all_t=pd.concat([df_all_t, df_t], axis=0)  # row  추가 
      
    # df.ffill() nan 일때 앞의 값으로 채운다 
    # df.bfill() nan 일때 뒤의 값으로 채운다 
    # df.fillna('') nan 일때 '' 로 채운다         
    # col1 에서 null일 때  앞의 값으로 채운다 
      
    col1 = df_all_t.iloc[0].ffill().fillna('').to_list()
    col2 = df_all_t.iloc[1].ffill().fillna('').to_list()        
    # c1, c2 중에 ''이 있으면 '_'을 넣지 않는다 
    df_all_t.columns=[c1 + c2  if c1=='' or c2 == '' else c1 + "_"+ c2 for c1, c2 in zip(col1, col2)]
    
      ### 3. 처음에 두개의 columns으로 사용한 row를 삭제한다
    df_all_t=df_all_t.drop(index=[0,1]) # df_all_t.iloc[0], df_all_t.iloc[1]  삭제
    
    df_all_t=df_all_t.ffill() 
    df_all_t=df_all_t.convert_dtypes()	# 자료마다 type변경을 한다
    all_df = df_all_t.select_dtypes(include='number') 
