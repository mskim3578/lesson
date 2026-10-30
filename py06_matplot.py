# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 11:08:55 2026

@author: letuin
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import openpyxl





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



x2 = np.linspace(0, 10, 10)  # 10을 포함한다
y2 = np.random.rand(10) 

plt.bar(x2, y2, color='skyblue', edgecolor='black')
plt.title("Bar Chart Example")
plt.xlabel("Category")
plt.ylabel("Values")
plt.show()
###---barh




# %%




x = np.linspace(0, 10, 10)
y = np.random.rand(10)
plt.barh(x, y, color='lightcoral')
plt.title("Horizontal Bar Chart Example")
plt.xlabel("Values")
plt.ylabel("Category")
plt.show()



x3 = np.linspace(0, 10, 100)
y3 = np.random.rand(100)
plt.scatter(x3, y3, color='green', marker='o')
plt.title("Scatter Plot Example")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()


import seaborn as sns
###--- heatmap
# sns.heatmap(data, 옵션)
data2 = np.random.rand(5,5)
data2
sns.heatmap(data2, annot=True, cmap='coolwarm')
plt.title("Heatmap Example")
plt.show()



###----box plot
# plt.boxplot(data, 옵션)
data = [np.random.randn(100), \
np.random.randn(100) + 2]
plt.boxplot(data, tick_labels=['Group 1', 'Group 2'], 
patch_artist=True)
plt.title("Box Plot Example")
plt.show()





data = np.random.choice(10, 100) +1
# bins=10 구간 개수 조정
plt.hist(data, bins=10, color='purple', 
edgecolor='black', alpha=0.7)
plt.title("Histogram Example")
plt.xlabel("Value")
plt.xticks(np.arange(1, 11))
plt.ylabel("Frequency")
plt.show()



###----Pie Chart
# plt.pie(values, labels, 옵션)
labels = ['A', 'B', 'C', 'D']
sizes = [30, 20, 35, 15]
# sizes = [12, 20, 35, 95]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', 
colors=['gold', 'lightblue', 'lightcoral', 'lightgreen'])
plt.title("Pie Chart Example")
plt.show()



###--- 여러개 그래프
# # fig, axes = plt.subplots(rows, cols)
fig, ax = plt.subplots(2, 2, figsize=(5, 5))
ax[0, 0].plot(x1, y1)
ax[0, 0].set_title("Line Plot")
ax[0, 1].bar(x2, y2)
ax[0, 1].set_title("Bar Chart")
ax[1, 0].hist(data, bins=10, color='purple', 
edgecolor='black', alpha=0.7)
ax[1, 0].set_title("Histogram")
ax[1, 1].scatter(x3, y3)
ax[1, 1].set_title("Scatter Plot")
plt.tight_layout()
plt.show()


x = np.linspace(0, 10, 10).astype(int)
y1 = np.random.rand(10)
y2 = np.random.rand(10)*100//10
y2=y2.astype(int)
plt.figure(figsize=(12, 6))
fig, ax1 = plt.subplots()
ax2=ax1.twinx()
ax1.bar(x, y1, color='b')
ax2.plot(x, y2, color='r')
ax1.set_ylabel('ax1', color='b')
ax2.set_ylabel('ax2', color='r')


ax1.set_xticks(np.arange(0, 11))
ax1.set_xticklabels(np.arange(0, 11), rotation=45) 
# <- 여기서 회전 설정!
ax2.tick_params(axis='y', labelcolor='r')
plt.title("Average Yield per HDP_DEPO (EQPID + Chamber)")
plt.tight_layout()
plt.show()
               
               

plt.figure(figsize=(12,8))   
chart_count=30
col=3
row=chart_count // col + 1
count=1
for i in range(chart_count):
   x = np.linspace(0, 10, 1000)
   y = np.random.rand(1000)
   plt.subplot(row, col, count)
   count +=1 
   plt.plot(x, y, color='blue', label=i)
   plt.title(f'{i} 번')
   plt.grid(True)

plt.legend()
plt.show()


################# matplot Ex
df = pd.read_excel("data/남북한발전전력량.xlsx")
df.info()

# 1 북한지역의 발전량만 조회. 
ndf = df.iloc[5:]
ndf.info()   

# 2. 전력량 (억kwh) column 지우기
del ndf['전력량 (억kwh)']
ndf['발전 전력별']
 
# 3. '발전 전력별'을 index로 전환
ndf=ndf.set_index("발전 전력별")

# 4. ndf 를 전치 행렬으로 수정 한다 
tdf=ndf.T
tdf.columns
    
# 5. 합계 ---> 총발전량 
tdf = tdf.rename(columns={'합계':'총발전량'})
tdf.head(10)    
tdf.info()
# 6. 전년도발전량 = 총발전량 - 1년
tdf['전년도발전량']=tdf['총발전량'].shift(1)

# 7. 증감율 컬럼 추가하기
# 증감율 :(현재-전년도)/전년도 * 100
#         (현재/전년도 - 1) * 100

tdf['증감율']=((tdf['총발전량']/tdf['전년도발전량'])-1)*100

# 좌측 y1 수력
# 우측 y2 증감율
import matplotlib.pyplot as plt
import numpy as np 
plt.rcParams['axes.unicode_minus'] = False
plt.rc("font",family="Malgun Gothic")



plt.figure(figsize=(12, 6))
fig, ax1 = plt.subplots()
ax2=ax1.twinx()
ax1.bar(tdf.index, tdf.수력, color='b', label="수력")
ax2.plot(tdf.index, tdf.증감율, color='r', label="증감율")
ax1.set_xlabel('연도')
ax1.set_ylabel('수력 발전량', color='b')
ax2.set_ylabel('전년 대비 증감율', color='r')
ax1.set_xticklabels(tdf.index, rotation=45, fontsize=8) 
# <- 여기서 회전 설정!
ax2.tick_params(axis='y', labelcolor='r')
plt.title("북한 전력 현황")
ax2.legend(loc='upper right')
ax1.legend(loc='upper left')
plt.tight_layout()
# plt.savefig("output/북한전력량.png", dpi=400, bbox_inches='tight')
plt.show()   


#자동차 연비데이터의 mpg 값을 히스토그램으로 출력하기
mpg = sns.load_dataset("mpg")
mpg.info() 
#  histogram
plt.hist(mpg.mpg, bins=10, color='purple', 
edgecolor='black', alpha=0.5, rwidth=0.8)
plt.xlabel('연비', rotation=45, fontsize=8)
plt.ylabel('수량', rotation=90, fontsize=8)
plt.show()


mpg["mpg"].plot(kind="hist", color='purple', 
edgecolor='black', alpha=0.5, rwidth=0.8)
plt.show()

###### scatter
plt.scatter(mpg["mpg"], mpg["weight"], color='green', marker='o')
plt.title("Scatter Plot Example")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

mpg.plot(kind='scatter', x='mpg', y='weight'
         , color='green', marker='o', s=10)
plt.show()

mpg['cylinders'].unique() 
    
mpg.plot(kind='scatter', x='mpg', y='weight'
         ,  marker='+', s=10, alpha=0.7,
         c=mpg['cylinders'], cmap='Blues'
         )
plt.show() 
mpg.info()    
######### pie  chart
df_origin=mpg.origin.value_counts()

df_origin.plot(kind='pie', autopct='%.1f%%',
               colors=['chocolate','bisque','cadetblue'],
               explode=[0.1, 0, 0],
               startangle=90,
               )
plt.show()    
 
###################
#  전세계 음주 데이터 분석하기 : drinks.csv

drinks = pd.read_csv("data/drinks.csv")
drinks.info()
'''
  country : 국가명
  beer_servings : 맥주소비량
  spirit_servings : 음료소비량
  wine_servings : 와인소비량   
  total_litres_of_pure_alcohol : 순수 알콜량
  continent : 대륙명
'''    
drinks.head() 
# 변수 = 컬럼 = 피처
# 상관계수 : 두연속적인 데이터의 상관관계 수치
# 피어슨 상관계수 : 기본.     
 
drinks[['beer_servings','wine_servings']].corr()  

# 켄달 상관계수 : 샘플 사이즈가 작은 경우.
#               동률데이터의 확율이 높은 경우  
drinks[['beer_servings','wine_servings']].corr(method='kendall') 
    
# 스피어만 상관계수 : 정규화가 되지 않는 데이터에 많이 사용 
drinks[['beer_servings','wine_servings']].corr(method='spearman') 

cols=drinks.columns[1:-1]
cols2=['beer','spirit','wine','alcohol']
corr=drinks[cols].corr()
sns.set(font_scale=1.5)
sns.heatmap(corr, cmap='Blues',
            cbar=True,
            annot=True,
            xticklabels=cols2,
            yticklabels=cols2,
            )

drinks.info()

drinks.isnull().sum()
drinks.continent.value_counts(dropna=False)
drinks.continent=drinks.continent.fillna("OT")

# 대륙별 국가의 갯수를 파이그래프로 출력하기
sns.set(font_scale=1)
#tolist() : 리스트로 형변환
labels = drinks['continent'].value_counts().index.tolist()
labels
#'AF', 'EU', 'AS', 'OT', 'OC', 'SA'
explode = (0, 0, 0, 0.1, 0, 0) 
plt.pie(drinks['continent'].value_counts(), #데이터값
    labels=labels,                     #라벨명. 대륙명
    autopct='%.0f%%',  #비율표시. %.0f : 소숫점이하 없음. %%:%문자
    explode=explode,  #파이의 위치지정. 0.1 : 1/10만큼 밖으로 표시
    shadow=True)
plt.title('null data to \'OT\'')
               
# 1 대륙별 total_litres_of_pure_alcohol 섭취량 평균

cont_mean=drinks.groupby("continent")\
    ['total_litres_of_pure_alcohol'].mean()

total_mean=drinks.total_litres_of_pure_alcohol.mean()

continents=cont_mean.index.tolist()
continents.append('Mean')

x_pos= np.arange(len(continents))
y=cont_mean.tolist()
y.append(total_mean)


bar_list=plt.bar(x_pos, y)
bar_list[len(continents)-1].set_color('r')
plt.xticks(x_pos, continents)
plt.show()
drinks.info()
#대한민국은 얼마나 술을 독하게 마시는 나라인가?
drinks.country.unique()

drinks["total_servings"] =\
    drinks["beer_servings"] + \
    drinks["spirit_servings"] +\
    drinks["wine_servings"]

#alcohol_rate : 알콜비율 (알콜섭취량/전체주류소비량) 추가
drinks["alcohol_rate"]=\
    drinks["total_litres_of_pure_alcohol"]/drinks["total_servings"]

drinks["alcohol_rate"].isnull().sum()
drinks["alcohol_rate"]=drinks["alcohol_rate"].fillna(0)

#alcohol_rate의 값으로 내림차순 정렬하기. alcohol_rate_rank 저장

alcohol_rate_df=drinks.sort_values(by='alcohol_rate', ascending=False)\
                       [['country','alcohol_rate']]


korea_rank=alcohol_rate_df.country.tolist().index('South Korea')

x_pos=np.arange(len(alcohol_rate_df.country.tolist()))
y=alcohol_rate_df.alcohol_rate.tolist()

bar_list=plt.bar(x_pos, y)
bar_list[korea_rank].set_color('r')

korea_alcohol_rate=\
 alcohol_rate_df[alcohol_rate_df['country']=='South Korea']['alcohol_rate'].values[0]


plt.annotate('South Korea : ' + str(korea_rank + 1)+"번째",   # 텍스트
          xy=(korea_rank, korea_alcohol_rate),                # 화살표가 가리키는 지점
          xytext=(korea_rank + 10, korea_alcohol_rate+0.1 ),  # 텍스트  위치할 지점
          arrowprops=dict(facecolor='red', shrink=0.05))      #속성을 정의하는 딕셔너리
            #shrink=0.05는 화살표의 양 끝을 5% 줄인다 
plt.show()







