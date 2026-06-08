# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:08:05 2026

@author: user
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams['axes.unicode_minus'] = False
plt.rc("font",family="Malgun Gothic")

#%%

def load_data():
    PATH_EQP="data/pjt2_eqp_paramters.xlsx"   
    PATH_COORD="data/pjt2_coordinate.xlsx" 
    PATH_THK="data/pjt2_thickness.xlsx" 
    
    df_all = pd.read_excel(PATH_EQP, engine='openpyxl')
    # df_all.columns
    # 1. df_all['chA_VIR_Step_Name']=='Depo'
    df_pecvd = df_all[df_all['chA_VIR_Step_Name']=='Depo']
    # 2. null있는 컬럼 삭제
    df_pecvd=df_pecvd.dropna(axis=1)
    
    # 3 ?
    # df_pecvd['chA_A0_mfc3_setpoint_SiH4']
    # [c for c in df_pecvd.columns if df_pecvd[c].nunique() <= 1]
    
    # 4. unique()가 1개인 컬럼을 삭제 한다 
    df_pecvd=df_pecvd.drop(columns=[c for c in df_pecvd.columns if df_pecvd[c].nunique() <= 1])
    
    #5 wafer df 만들기 
    df_coor = pd.read_excel(PATH_COORD, engine='openpyxl')
    df_thk = pd.read_excel(PATH_THK, engine='openpyxl')
    
    # 6.컬럼명 공백 제거
    df_thk.columns=[s.strip() for s in df_thk.columns]
    
    # 7. 좌표와 두께 파일 결합
    df_waf = pd.concat([df_coor, df_thk.iloc[:, 1:]], axis=1)   
    
    return df_pecvd, df_waf
    
'''
df_pecvd, df_waf=load_data()
df_pecvd.info()
'''

















# %%


def sample_load_data():
    drinks = pd.read_csv("data/drinks.csv")
    corr1 = drinks[['beer_servings', 'wine_servings']] \
             .corr()
    cols=drinks.columns[1:-1]
    corr1=drinks[cols].corr()

    sns.heatmap(corr1, cmap="Blues", 
                fmt='f', 
                annot=True,
                # cbar=False,
                linewidth=3)
    # plt.show()
    return plt   #app2 에서 web에 print 함
    
sample_load_data()
