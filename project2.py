# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 09:54:37 2025

@author: letuin
"""
# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import glob as glob  

'''



'''

def read_glob(dirform):
    global folderlist, df_data, file_count
    
    file_count=0
    df_data = pd.DataFrame([])    
    folderlist=glob.glob(dirform, recursive=True)
    for f_folder in folderlist:
        file_count +=1
        one_df=read_file(f_folder)
        df_data = pd.concat([df_data, one_df], ignore_index=None)
        
def read_file(filename):
    # 1. 한개의 파일 읽기
    global col
    read_data = pd.read_csv(filename, header=None)   
    lotid=read_data.iloc[9][2]     # lotid
    rdate=read_data.iloc[12][2]   
    rdate=rdate.split()[0]    # date
    rrate=read_data.iloc[2][7]  # rate
    print(lotid, rdate, rrate)
    # 2 분석자료 분리
    one_data=read_data.iloc[27:]
    
    # 3 column 정리
    col1=read_data.iloc[16][7:].to_list()
    col2=read_data.iloc[17][7:].to_list()
    col=[x.strip() + "_"+ y.strip()  for x, y in zip(col1, col2)]
    col3 = [' Test No.', ' Coordinate', ' Soft Bin', ' Hard Bin', ' Time[s]', ' Result', ' Fail Item'] 
    col = [x.strip() for x in col3] +col
    one_data.columns=col
    
    # 4 데이터 추가
    one_data['rdate']=rdate
    one_data['lotid']=lotid
    rrate.replace("%",'')
    one_data['rrate']=rrate
    
    one_data['row']=[set_coor(x)[1] for x in one_data['Coordinate']] # x,y값 수정
    one_data['col']=[set_coor(x)[0] for x in one_data['Coordinate']] # x,y값 수정
  
    one_data['Fail Item']=[x.strip() for x in one_data['Fail Item']] 
    one_data['count']=[ 0 if "'" in x else 1 for x in one_data['Fail Item']]
    
    
    
    
    return one_data 
# %%

def set_coor(coor):
    coor = str(coor)
    coor = coor.replace('X', '')
    coor = coor.replace('Y', '')
    coor = coor.split('_')   
    coor = [ int(x.strip()) for x in coor]
    # print(coor)
    return coor




# lot별 fail 현황
def lot_fail_one(lots, spro):
    
    for lot in lots:
        groups = df_data[df_data['lotid']==lot].groupby('Fail Item')   
        temp=df_data[df_data['lotid']==lot]
        temp2=temp[temp['Fail Item']=="'"]
        pro=len(temp2)/len(temp)*100
        plt.figure()
        plt.clf()
        for name, group in groups:
               namelabel = name+"="+str(len(group))       
               color=colors[fail_item.index(name)]
               # print(namelabel, len(group), color)
               if name=="'":
                plt.scatter(group.col, group.row, label=namelabel,  color="lightcyan",
                               s=30, marker='s')
               else :
                plt.scatter(group.col, group.row, label=namelabel,  
                            color=color,s=30, marker='s')       
              


        plt.xticks(range(-8, 12))
        if pro <= spro:
            plt.title( f"{lot} :{pro:.2f}% 불량", color='red')
        else:
            plt.title( f"{lot} :{pro:.2f}%  우량")
        plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
        plt.axvline(0, color='black', linestyle='--', linewidth=0.5)
        plt.legend(fontsize=7)
        # plt.axis("off")
        # plt.grid()   #바탕화면에 모눈 표시 한다 
        plt.show()


def lot_fail_onepage(lots, spro):
    row= len(lots)//3 if len(lots) % 3 == 0 else len(lots)//3+1
    #plt.figure(figsize=(5, 5), num='전체 lotid')
    #plt.clf()
    count=1
    for lot in lots:
        temp=df_data[df_data['lotid']==lot]
        temp2=temp[temp['Fail Item']=="'"]
        pro=len(temp2)/len(temp)*100
        
        groups = df_data[df_data['lotid']==lot].groupby('Fail Item')   
        plt.subplot(row, 3, count)  
        count +=1
        for name, group in groups:
               #namelabel = name+"="+str(len(group))     
               namelabel = name+"="+str(len(group))       
               color=colors[fail_item.index(name)]
               # print(namelabel, len(group), color)
               if name=="'":
                plt.scatter(group.col, group.row, label=namelabel,  color="lightcyan",
                               s=30, marker='s')
               else :
                plt.scatter(group.col, group.row, label=namelabel,  
                            color=color,s=30, marker='s')       
              


        plt.xticks(range(-8, 12))
        if pro <= spro:
            plt.title( f"{lot} :{pro:.2f}% 불량", color='red', fontsize=7 )
        else:
            plt.title( f"{lot} :{pro:.2f}%  우량" , fontsize=7 )
        
        plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
        plt.axvline(0, color='black', linestyle='--', linewidth=0.5)
        #plt.legend(fontsize=7)
        plt.axis("off")
    plt.show()
        

#### 날자별 heatmap : 여러개의 lotid나, 날자별 error를 count 해야한다  

def fail_heatmap_onepage(datas, key, annot=False):
    
    row= len(datas)//3 if len(datas) % 3 == 0 else len(datas)//3+1
   
    #  plt.subplots(row, 3, figsize=(5, 5))
    plt.figure(figsize=(10, 10), num=key.capitalize()+' 리스트')
    plt.clf()
    mcount=1
    for dd in datas:
       
        date_df=df_data[df_data[key]==dd]
        #groups = df_data[df_data['rdate']==dd].groupby('Fail Item')   
        plt.subplot(row, 3, mcount)  
        mcount +=1 
        df_heat=date_df.pivot_table(
                            index='row',
                            columns='col', 
                            aggfunc='sum',  # 각 row, col에 해당하는 Fail Item의 values_count를 한다
                            values='count')  #count의 값은 0,1이 저장되어 있음
        sns.heatmap(df_heat,cmap="Greens",annot=annot)     
        #plt.title(dd+" "+key)
        plt.title(dd)
      
    plt.tight_layout()
    plt.show() 
########################  function end



    
# %%   
'''
1) Fail_Item
2) 날자별
3) lotid
'''
 
read_glob('data2/*/*/*.csv')   
df_data.columns
df_data.info()
# df_data (511650, 29)    # row, col 추가 한다 



#%%

cmap=plt.get_cmap('tab20')  # 
colors = [cmap(i) for i in range(cmap.N)]  #20개의 color

# color분류 
["'",
 'T5_VTH1',
 'T7_VTH_DEL1',
 'T17_VTH_DEL2',
 'T16_VTH3', 
 'T10_BVDSS1',
 'T11_BVDSS2',
 'T12_BVDSS_DEL1', 
 'T4_IGSSR2',
 'T15_IGSSR4', 
 'T3_IGSSF1',
 'T14_IGSSF3',
 'T2_CONT',
 'T13_IDSS1' 
 ]


plt.rcParams['axes.unicode_minus'] = False  #chart에 마이너스 프린트 
plt.rc("font",family="Malgun Gothic")   # 한글 프린트   



# %%

        
######   lot별 시각화 Fail item 시각화 (%)  97%이상을 우량이 라고 가정한다
fail_item = list(df_data['Fail Item'].unique())  
lot_fail_one(df_data['lotid'].unique().tolist()[:9], 97)
lot_fail_onepage(df_data['lotid'].unique().tolist()[:9],97)



# 전체 자료기반 
# fail item [1:14]  우량 자료는 표시 하지 않음
# lotid, rdate는 9개씩 프린트 한다

fail_heatmap_onepage(df_data['lotid'].unique().tolist()[:9],'lotid', False)
fail_heatmap_onepage(df_data['rdate'].unique().tolist()[:9],'rdate')
fail_heatmap_onepage(df_data['Fail Item'].unique().tolist(),'Fail Item', False)  #fail 전체 가능함




