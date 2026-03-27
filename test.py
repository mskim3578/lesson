import pandas as pd

import matplotlib.pyplot as plt 
import numpy as np
plt.rcParams['font.family'] = 'Malgun Gothic'   # Windows


# 마이너스 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False


def load_data(stepname='Depo'):
    global df_stepname, df_waf
    
    # df_stepname      
    df_all = pd.read_excel("data/project2_eqp_paramters.xlsx")
    
    df_stepname = df_all[df_all['chA_VIR_Step_Name']==stepname]
    df_stepname.dropna(axis=1, inplace=True)  # null이 있는 column을 삭제 한다     
    df_stepname = df_stepname[df_stepname["chA_AO_mfc2_setpoint_Si2H6"] != 0]
    
    # df_waf
    df_coor = pd.read_excel("data/project2_coordinate.xlsx")
    df_thk=pd.read_excel('data/project2_thickness.xlsx' ,sheet_name='thickness')    
    df_thk.columns=[s.strip() for s in df_thk.columns]  # ' Run#1'
    df_waf=pd.concat([df_coor,df_thk[df_thk.columns[1:]] ], axis=1) # column으로 합한다
 
    return df_stepname, df_waf


def sample_scatter():  
    fig=plt.figure(figsize=(10, 8))
    x1 = np.linspace(0, 10, 100)  # 10포함 한다
    y1 = np.random.rand(100)   
    
    # plt.plot(x1, y1, color='b', linestyle='-', marker='o', label="plt.plot()")
    plt.plot(x1 , y1, color='b', linestyle='-', label="plt.plot()")
    plt.title("Line Plot 그래프")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.yticks([x/10 for x in range(11)])
    plt.legend()
    plt.show()

    # plt.show()    
    return fig
