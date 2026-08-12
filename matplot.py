# %%
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['axes.unicode_minus'] = False
plt.rc("font",family="Malgun Gothic")

# %%
# plt.plot(x, y, 스타일)
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

# %%


import tensorflow as tf

x2 = np.linspace(0, 10, 10)  # 10을 포함한다
y2 = np.random.rand(10) 

plt.bar(x2, y2, color='skyblue', edgecolor='black')
plt.title("Bar Chart Example")
plt.xlabel("Category")
plt.ylabel("Values")
plt.show()
###---barh
# %%
x4 = np.ceil(np.linspace(0, 10, 10))
y4 = np.ceil(np.random.rand(10)*10)

plt.barh(x4, y4, color='lightcoral')
plt.title("Horizontal Bar Chart Example")
plt.xlabel("Values")
plt.ylabel("Category")
plt.show()



# %%
x3 = np.linspace(0, 10, 100)
y3 = np.random.rand(100)


plt.scatter(x3, y3, color='green', marker='o')
plt.title("Scatter Plot Example")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()


# %%

plt.plot(x1 , y1, color='b', linestyle='-', label="plot")
plt.bar(x2, y2, color='skyblue', edgecolor='black', label="bar")
plt.scatter(x3, y3, color='green', marker='o', label='scatter')
plt.title("Multi Example")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend(loc="best")
plt.show()

# %%
###--- heatmap
# sns.heatmap(data, 옵션)
import seaborn as sns
npint = np.random.randint(0, 100, (10,10))
data2 = np.random.rand(5,5)
data2
sns.heatmap(data2, annot=True, cmap='coolwarm')
plt.title("Heatmap Example")
plt.show()

###----box plot
#  plt.boxplot(data, 옵션)
#  np.random.randn(100)  np.random.randn → 표준 정규분포를 따르는 난수를 생성
#  patch_artist=True box내부의 color를 채운다

data = [np.random.rand(100)*100, \
                 np.random.rand(100)*100 + 2, \
                 np.random.rand(100)*100 + 4]

plt.boxplot(data, tick_labels=['Group 1', 'Group 2', 'Group 3'],
                                      patch_artist=True)
plt.title("Box Plot Example")
plt.show()

###----Pie Chart
#  plt.pie(values, labels, 옵션)

labels = ['A', 'B', 'C', 'D']
sizes = [30, 20, 35, 15]

plt.pie(sizes, labels=labels, autopct='%1.1f%%', 
      colors=['gold', 'lightblue', 'lightcoral', 'lightgreen'])
plt.title("Pie Chart Example")
plt.show()



data = np.random.choice(10, 100) + 1
# #  fig, axes = plt.subplots(rows, cols)
fig, ax = plt.subplots(2, 2, figsize=(5, 5))
ax[0, 0].plot(x1, y1)
ax[0, 0].set_title("Line Plot")

ax[0, 1].bar(x2, y2)
ax[0, 1].set_title("Bar Chart")

ax[1, 0].hist(data, bins=10, color='purple', edgecolor='black', alpha=0.7)
ax[1, 0].set_title("Histogram")

ax[1, 1].scatter(x3, y3)
ax[1, 1].set_title("Scatter Plot")

plt.tight_layout()
plt.show()


###---2) 여러개 그래프
data = np.random.choice(10, 100) + 1

plt.figure(figsize=(5, 5))
plt.subplot(2,2,1)
plt.plot(x1, y1)
plt.title("Line Plot")

plt.subplot(2,2,2)
plt.bar(x2, y2)
plt.title("Bar Chart")

plt.subplot(2,2,3)
plt.hist(data, bins=10, color='purple', edgecolor='black', alpha=0.7)
plt.title("Histogram")

plt.subplot(2,2,4)
plt.scatter(x3, y3)
plt.title("Scatter Plot")

plt.tight_layout()
plt.show()

###---3) 여러개 그래프
# 전체 도화지(Figure) 크기 설정
plt.figure(figsize=(12, 8))
count=1
chart_count=10
col = 3
row = chart_count // col + 1

for i in range(chart_count):
    x = np.linspace(0, 10, 1000)
    y = np.random.rand(1000)
    plt.subplot(row, col, count)
   
    count +=1
    plt.plot(x, y, color='blue', label=i)
    #plt.scatter(x, y, color='blue', label=i)
    plt.title('Sine Wave')
    plt.grid(True)
plt.legend()
plt.show()




# %%
####  2개의 y구간 정의 그래프 그리기


x = np.linspace(0, 10, 10).astype(int)
y1 = np.random.rand(10)
y2 = np.random.rand(10)*100//10
y2=y2.astype(int)

plt.figure(figsize=(12, 6))
fig, ax1 = plt.subplots()
ax2=ax1.twinx()


ax1.bar(x, y1)
ax2.plot(x, y2, color='r')  ###  2

ax1.set_ylabel('ax1', color='b')
ax2.set_ylabel('ax2', color='r')

ax1.tick_params(axis='y', labelcolor='b') ### 1

# ax1.set_ylim(0, 160)  # ax1  y좌표의 구간을 표시한다 
ax1.set_xticks(x)
ax1.set_xticklabels(x, rotation=45)  # <- 여기서 회전 설정!
#   y축의 좌표값의 color를 red로 수정 하는 방법
ax2.tick_params(axis='y', labelcolor='r')
#ax2.set_ylim(0, 1.75) # ax2 y좌표의 구간을 표시한다 

plt.title("Average Yield per HDP_DEPO (EQPID + Chamber)")
plt.tight_layout()
plt.show()


#   여러개 그래프
import statistics
plt.figure(figsize=(12,8))
chart_count=10
col=3
row=chart_count//col+1
count=1
for i in range(chart_count):
    x=np.linspace(0, 10, 500)
    y=np.random.rand(500)
    tmean= sum(y) / len(y)
    tstd = statistics.stdev(y)
    
    plt.subplot(row, col, count)
    plt.axhline(y=tmean, color='blue')
    plt.axhline(y=(tmean+(tstd*3)), color='r')
    plt.axhline(y=(tmean-(tstd*3)), color='r')
    count +=1
    plt.plot(x,y, color='blue', label=i)
    plt.title(f'EDA 데이터 탐색 - {i}')
    plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()







########################  matplot ex
import pandas as pd
df_age=pd.read_csv('data_folder/age.csv', encoding='cp949',
                   thousands=",", index_col=0)

# 1. columns 확인
df_age.columns
cols_10= df_age.columns[3:13]
df_age.info()

df_10=df_age[cols_10]
df_10.info()




#2.  모든컬럼의 내용을  1세, 2세..... 로 수정 하세요
new_col = [ x.split('_')[2] for x in cols_10]
df_10.columns=new_col

plt.bar(range(10), df_10.sum())
plt.show()

#3. x : 1 ~10, x col

'''
<class 'pandas.DataFrame'>
Index: 3910 entries, 서울특별시  (1100000000) to 제주특별자치도 서귀포시 예래동(5013062000)
Columns: 103 entries, 2025년06월_계_총인구수 to 2025년06월_계_100세 이상
dtypes: int64(103)
memory usage: 3.3 MB
'''

# pip install  openpyxl
# 남북한 발전전력량
df_sn=pd.read_excel('data_folder/남북한발전전력량.xlsx')
df_sn.info()
df_sn.head(10)
# 북한지역의 발전량만 조회
df_n=df_sn.iloc[5:]
df_n.columns

del df_n['전력량 (억kwh)']

df_n.set_index("발전 전력별", inplace=True)
df_n.info()

df_t = df_n.T
df_t.info()
df_t.head(10)
df_t=df_t.rename(columns={'합계':'총발전량'})

df_t['전년도발전량']=df_t['총발전량'].shift(1)
df_t['증감율']=(df_t['총발전량']-df_t['전년도발전량'])/df_t['전년도발전량']*100









