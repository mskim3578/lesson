# -*- coding: utf-8 -*-

# %%

import openpyxl 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

#  렛유인 project1 
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
   
    

def all_graph(n_df, allview=True, sigma=2, graph_type='plot', figurename=None ):
  
  plt.figure(figsize=(10, 10), num=figurename)
  count=1
  row_count = len(n_df.columns)//6 + 1 #몫 더하기 1을 한다  
  labelname1 = "sigma-"+str(sigma)
  labelname2 = "sigma+"+str(sigma)  
  plt.suptitle(fname+" : "+sheetname[sheetindex])
  
  for col in n_df.columns :
    
    xx= n_df[col].values
    df_mean = n_df[col].mean()
    df_std = n_df[col].std()
    
    
   
    #print(col, df_std, df_mean)
    x = range(len(xx)) 
    
    upperdf= n_df[n_df[col]>(df_mean + (df_std*sigma))]
    lowerdf= n_df[n_df[col]<(df_mean - (df_std*sigma))]
   
    
    
    # df[col].fillna(df_mean) # 확인한다 !!!!!!!!!
    
    # sigma - 2,3 보다 큰 자료가 있거나, allview 가 True 일때 
    if len(upperdf) != 0 or len(lowerdf) != 0 or allview:
      ax = plt.subplot(row_count, 6, count)
      
      count +=1
      if graph_type == 'hist':
        plt.hist(xx)
      elif graph_type == 'scatter':
        plt.plot(x, [df_mean - (sigma * df_std) for x in range(len(xx))], label=labelname1)   
        plt.plot(x, [df_mean + (sigma * df_std) for x in range(len(xx))], label=labelname2)
        plt.scatter(x, xx)
      else:
        plt.plot(x, [df_mean - (sigma * df_std) for x in range(len(xx))], label=labelname1)   
        plt.plot(x, [df_mean + (sigma * df_std) for x in range(len(xx))], label=labelname2)
        plt.plot(x,xx,  color='b', linestyle='-')  
       
      #plt.legend()  
     
      plt.title(f"{col}", fontsize=8)
      plt.axis("off")  #x, y 좌표 프린트 않한다   
      
    
  plt.tight_layout()  #graph 자동 조정
  plt.show(block=True)         


def one_graph(n_df, index, sigma=2, figurename="one_chart"):
 
  xx= n_df[n_df.columns[index]].values
  x = range(len(xx)) 
  df_mean = n_df[n_df.columns[index]].mean()
  df_std = n_df[n_df.columns[index]].std()
  labelname1 = "sigma-"+str(sigma)
  labelname2 = "sigma+"+str(sigma)  
  plt.figure(figsize=(10, 10), num=figurename)  
  plt.plot(x, [df_mean - (sigma * df_std) for x in range(len(xx))], label=labelname1)   
  plt.plot(x, [df_mean + (sigma * df_std) for x in range(len(xx))], label=labelname2)
  plt.plot(x,xx,  color='b', linestyle='-')  
  plt.title(f"{n_df.columns[index]}", fontsize=8)  
  plt.show()         


# df1은 좌측 y 기준의 그래프, df2는 우측 y 기준의 그래프
# page별로 복수의 chart를 한개의 figure에 프린트 하는 방법



def twins_chart(df1,  df2,  sheetindex):    
    #1. color list
    colors=["blue","green","red","cyan","magenta","yellow","black"]
    col_index=0
    #2. x좌표의 range 혹은 값
    x = range(1000)
    
    #3. twinx() 위한 figure
    fig, ax1 = plt.subplots(figsize=(10, 6)) # figsize로 그래프 크기 조절
    ax1.set_xlabel('range(1000)') # X축 레이블
   
    
    #ax1.tick_params(axis='y', labelcolor=colors[col_index]) # 왼쪽 Y축 눈금 레이블 색상
    #4.  왼쪽 Y축 레이블 및 색상
    ax1.set_ylabel(df1.columns.to_list(), color=colors[col_index]) 
    
    #5 dataframe에 column별로 그린다
    # left 폭 확인
    left_max=0
    left_min=sys.maxsize    
    for col1 in df1.columns:        
       if df1[col1].max()>left_max : left_max = df1[col1].max() 
       if df1[col1].min()<left_min : left_min = df1[col1].min()
    
    # 차트 폭조정
    print(left_max,left_min)
    left=(left_max-left_min)/3 
    
    ax1.set_ylim(left_min-left, left_max+left)    
   
    
    for col1 in df1.columns:
       ax1.plot(x, df1[col1], color=colors[col_index], label=col1) 
       # color change 한다
       col_index +=1
      
    # 6. twinx()를 사용하여 두 번째 Axes (ax2) 생성 (X축 공유, 새 Y축 생성)
    # top 그래프는  있는 그대로 표현한다  
    ax2 = ax1.twinx()
  
    # right
    #ax2.tick_params(axis='y', labelcolor=colors[col_index]) # 왼쪽 Y축 눈금 레이블 색상
    right_max=0
    right_min=sys.maxsize
   
    ax2.set_ylabel(df2.columns.to_list(), color=colors[col_index]) # 왼쪽 Y축 레이블 및 색상
   
    for col2 in df2.columns:  
       if df2[col2].max()>right_max : right_max = df2[col2].max() 
       if df2[col2].min()<right_min : right_min = df2[col2].min()   
    
    # 차트 폭조정
    right=(right_max-right_min)/4
    print(right_max,right_min, right)
    ax2.set_ylim(right_min, right_max*3) 
       
    for col2 in df2.columns:  
       ax2.bar(x, df2[col2], color=colors[col_index], label=col2) # 데이터 플로팅  
       col_index +=1
    
    
    # 7. 그래프의 제목 설정
    fig.suptitle(fname+"/"+sheetname[sheetindex])
    # 8 왼쪽 y 구간 설정 
    
    # 9 오른쪽 Y축 
    # 예시: 습도를 40%에서 80% 사이로 제한   
    
    # 10 legend 설정 
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # 11 자동 layout 조정
    fig.tight_layout()
    plt.show()      
 
    
def all_corr(type_df, title_name):
  corrdf = type_df.corr()
  sns.heatmap(corrdf, annot=False, cmap='coolwarm')
  plt.title(title_name)
  plt.show()
     
def twocolumn_chart(df, li, chart):
    colors=["blue","green","red","cyan","magenta","yellow","black"]
    col_index=1
    if chart == 'scatter':
        plt.scatter(df[li[0]], df[li[1]], color=colors[col_index]) # 데이터 플로팅  
    else:
        plt.plot(df[li[0]], df[li[1]], color=colors[col_index]) # 데이터 플로팅 
    plt.title(li[0]+" & "+ li[1]) 
    plt.xlabel(li[0])     
    plt.ylabel(li[1])
    plt.show()
      
    
   
################### function end         
# %%    load data & preprocess

#####  공정자료의 데이터 분석 입니다 이자료는 소자

# 1. set_loaddata  :  excel 파일 번호 선택
excel_num="5"   # data1/Raw data_ref_C_EX1.xlsx
sheetindex=1    # sheetname : ['Yield', 'ET(DC)', 'ADI_ACI CD', 'Thickness', 'FDC', 'Item 설명']

set_loaddata("data1/Raw data_ref_C_EX"+excel_num+".xlsx")  #한개의 excel file을 읽어서 sheet별로 작업한다
pre_process(sheetindex)  # 전처리 
print(sheetname)
print(number_df.columns)

# %% 
######################################    chart

# 2. 전체 자료중에 이상치를 확인 하는 작업을 한다 
all_graph(number_df, sigma=2, allview=False, figurename=sheetname[sheetindex] ) 
all_graph(number_df, sigma=2, figurename=(sheetname[sheetindex]+" = number all" ) )

# 3. 전체 feature간 상관계수
all_corr(number_df, sheetname[sheetindex])


# 4. 이상치를 확인후에 한개의 차트
one_graph(number_df, 1)

list_col=number_df.columns.to_list()

# 두게의 


# 5 twin chart 두개의 chart를 한번에 표현 한다 
twins_chart(number_df[["BIN1_Prime"]], number_df[['BIN20_Short','BIN19_Open']], sheetindex) 
twins_chart(number_df[["BIN1_Prime"]], number_df[['BIN19_Open','BIN20_Short']], sheetindex) 



#   number_df[[list_col[1]]] 은 dataframe이여야 한다 
twins_chart(number_df[[list_col[1]]], number_df[[list_col[3]]], sheetindex)
twins_chart(number_df[["BIN1_Prime",'BIN19_Open']], number_df[['BIN20_Short']], sheetindex)    


# 두개의 컬럼을 x, y로 연결 scatter혹은 line 차트를 그린다 


    
# 6   특정 sheet name별 2개의 feature (x,y) 분석
twocolumn_chart(number_df, ['VthN_S_[V]','IdoffP_S_[㎁/㎛]' ],'scatter') 
    
