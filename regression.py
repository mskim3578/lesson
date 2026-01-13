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

# 고상관 피처 자동 제거
def highcorr_pro(target, threshold):
    # 상관계수가 0.9 이상인 피처들 중 하나를 삭제
   
    tdf = all_df.drop(columns=[target])  # target은 정리 되면 않된다
    corr_matrix = tdf.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # 삭제할 컬럼 이름 추출
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    print(f"#### highcorr_pro: 삭제될 고상관 피처 개수: {len(to_drop)} ")
    return to_drop
   

# target과 상관계수가 임계점 이하인 피쳐     
def targetcorr_pro(target, low_threshold):
    # Target 변수와의 상관계수만 추출하여 정렬
    corr_matrix = all_df.corr().abs()
    
    target_corr = corr_matrix[target].sort_values(ascending=False)     # target과의 상관계수
    # 특정 임계값(예: 0.1) 이상의 피처만 선택
    to_drop = target_corr[target_corr < low_threshold].index.tolist()
    print(f"#### targetcorr_pro: 삭제될 저연관 피처 개수: {len(to_drop)} ")
    return to_drop
    

def select_target(target, highcorr, targetcorr):  
    global all_df, model_df, all_col, tdf
    all_col=  all_df.columns.to_list()   
    to_drop=set()
    #  다중공선성 제거
    to_drop.update(highcorr_pro(target, highcorr))  
    to_drop.update(targetcorr_pro(target, targetcorr))
    to_select=list(set(all_col).difference(to_drop))
    
    print(f'{target} --->  선택 피쳐 : {to_select}')
    ### 3  예측
    tdf=all_df[to_select]
    #   XGBoost  column name에 [,],<가 있으면 않된다 
    tdf.columns = [
        str(col).replace('[', '_').replace(']', '_').replace('<', '_') 
        for col in tdf.columns
    ]
    
    target=target.replace('[', '_').replace(']', '_').replace('<', '_') 
    # 전체 r2 chart
    model_rep=[]
    for regmodel in models:    
        mse, r2=predict_pro(tdf, target, regmodel)
        model_rep.append([regmodel,mse,r2])
        


   

    # 1. 그래프 그리기
    model_df=pd.DataFrame(model_rep, columns=['model','mse', 'r2'])
    model_df=model_df.sort_values(by='r2', ascending=False)
    x_range=range(len(models))
    labels=[ x for x in models]
    plt.figure(figsize=(10, 6))
    # plt.plot(x_range, model_df['r2'].values, marker='o', linestyle='-', color='b')
    plt.bar(x_range, model_df['r2'].values, color='b', edgecolor='black', linewidth=0.5, width=0.7)
    # 2. x축 눈금 설정
    plt.xticks(x_range, labels, rotation=45, ha='right')
    plt.ylim(model_df['r2'].min()-0.1, model_df['r2'].max()+0.1)
    # 3. 핵심: 직선 위에 r[0](모델 이름) 또는 r[1](값) 표시하기
    for i, r in enumerate(model_df['r2'].to_list()):
        # plt.text(x좌표, y좌표, 출력할내용)
        # y좌표에 약간의 오프셋(예: +0.01)을 주면 점 바로 위에 글자가 뜹니다.
        plt.text(i, r + 0.002, f'{r:.4f}  ', 
                 fontsize=9, 
                 ha='center',    # 가로 정렬: 중앙
                 va='bottom',    # 세로 정렬: 하단 (점 위에 떠 있게 함)
                 fontweight='bold')

    plt.title(f"{target} R2 Scores")
    plt.xlabel("Models")
    plt.ylabel("R2 Score")
    plt.tight_layout()
    plt.show()


  
