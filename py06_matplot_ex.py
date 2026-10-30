# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 10:19:07 2026

@author: letuin
"""


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
plt.rcParams['axes.unicode_minus'] = False
plt.rc("font",family="Malgun Gothic")
import openpyxl

# python.exe -m pip install --upgrade pip
# pip install openpyxl


###  연합 그래프 작성
df = pd.read_excel("data/남북한발전전력량.xlsx")
df.info()
df
#북한지역의 발전량만 조회.
df = df.loc[5:]
df
df.info()

del df["전력량 (억kwh)"]
df.set_index("발전 전력별", inplace=True)
df=df.T
df = df.rename(columns={"합계":"총발전량"})
df.info()
#총발전량-1년 추가 :전년도 발전량
df.head()
df["전년도발전량"] = df["총발전량"].shift(1)
df.head()
#증감율 컬럼 추가하기
# 증감율 :(현재-전년도)/전년도 * 100
#         (현재/전년도 - 1) * 100
df["증감율"]=((df["총발전량"]/df["전년도발전량"]) - 1) * 100
df.head()

#ax1 = df[['수력']].plot(kind='bar', figsize=(20, 10), width=0.7) 

fig, ax1 = plt.subplots()
ax1.bar(df.index, df.수력) 

ax2 = ax1.twinx()
ax2.plot(df.index, df.증감율, ls='--', marker='o', markersize=10, 
        color='green', label='전년대비 증감율(%)')
ax1.set_ylim(0, 200) #막대그래프 y축의 값의 범위
ax2.set_ylim(-50, 50)#선그래프  y축의 값의 범위  
ax1.set_xlabel('연도', size=20) #x축값의 설명
ax1.set_ylabel('발전량(억 KWh)') #막대그래프 y축값의 설명
ax2.set_ylabel('전년 대비 증감율(%)') #선그래프 y축값의 설명
plt.title('북한 전력 발전량 (1990 ~ 2016)', size=30) #전체 그래프 제목
ax1.legend(loc='upper left') #범례: 왼쪽 위 위치
ax2.legend(loc='upper right') #범례: 오른쪽 위 위치
plt.show() #화면에 표시.



plt.figure(figsize=(12, 6))
fig, ax1 = plt.subplots()
ax1.bar(df.index, df.수력)
ax1.set_ylabel('ax1', color='b')
#ax1.set_ylim(0, 160) 
ax1.set_xticks(df.index)
ax1.set_xticklabels(df.index, rotation=45)  # <- 여기서 회전 설정!
ax2=ax1.twinx()

ax2.plot(df.index, df.증감율)
ax2.tick_params(axis='y', labelcolor='r')
ax1.set_ylabel('ax2', color='b')
#ax2.set_ylim(0, 1.75) 

plt.title("Average Yield per HDP_DEPO (EQPID + Chamber)")

plt.tight_layout()
plt.show()



#그래프를 이미지파일로 저장.
#savefig(저장파일명,해상도,이미지크기설정)
plt.savefig("img/북한전력량.png", dpi=400, bbox_inches="tight")
plt.show() #화면에 표시.

#자동차 연비데이터의 mpg 값을 히스토그램으로 출력하기
df = sns.load_dataset("mpg")
df.info()
#DataFrame plot 히스토그램 출력
df["mpg"].plot(kind="hist")
plt.show()

#간격을 20개로 분리한 히스토그램 출력


df["mpg"].plot(kind="hist",bins=20,color='coral',\
               figsize=(10,5),histtype='bar', width=1)
plt.title("MPG 히스토그램")
plt.xlabel("mpg(연비)")
plt.show()

df["mpg"].min()  # 9.0
df["mpg"].max()  # 46.6


#weight,mpg 데이터의 산점도 출력하기
#DataFrame.plot(kind="scatter") : 그래프 종류
# x='mpg' : x축의 사용될 컬럼명
# y='weight' : y축의 사용될 컬럼명
# s=50 : 점의 크기지정
df.plot(kind='scatter', x='mpg', y='weight', c='coral',\
        s=50,figsize=(10, 5))
    
#matplot모듈을 이용하여 산점도 출력
plt.figure(figsize=(10,5)) #새로운 그래프창 
#scatter(x축데이터,y축데이터) : 그래프 종류 
#df["mpg"] : x축데이터
#df["weight"] : y축데이터
plt.scatter(df["mpg"],df["weight"],c="coral",s=20)
plt.show()


# 색상으로 데이터 값을 설정.
# marker="+" : 산점도 점의 모양
#cmap='magma' : matplot 모듈에서 숫자에 해당하는 색의 목록
#       viridis,inferno,magma,cividis,...
#c=df["cylinders"] : cylinders의 값에 해당하는 색을 cmap에서 선택 

# x="weight":df["weight"], y="mpg"
df.plot(kind="scatter",x="weight",y="mpg",marker="+", figsize=(10,5),\
         cmap='magma',c=df["cylinders"],s=50,alpha=0.7)   
plt.title("산점도:mpg-weight-cylinders")
plt.show()

c_size=df["cylinders"]/df["cylinders"].max()*100
c_size=df["cylinders"]*10

df.plot(kind='scatter', x='mpg', y='weight', c='coral',\
        s=c_size,figsize=(10, 5))
plt.title("산점도:mpg-weight-cylinders")
plt.show()


#파이그래프
#origin 컬럼 : 제조국.[usa,japan, europe]
df_origin = df.origin.value_counts()
df_origin
type(df_origin)
df_origin.plot(kind="pie") #파이그래프 
plt.title("자동차 생산국",size=20)
plt.legend(labels=df_origin.index,loc="upper left")   
plt.show()
'''
autopct="%.1f%%" : 파이그래프에 비율 표시
      %.1f : 소숫점이하1자리로표시
      %%   : % 문자

startangle=90 : 기본설정 위치에서 90도로 시작위치 변경
      
'''
df_origin.plot(kind="pie",figsize=(7,5),autopct="%.1f%%",startangle=90,
               colors=['chocolate','bisque','cadetblue']) #파이그래프 
plt.title("자동차 생산국",size=20)
plt.legend(labels=df_origin.index,loc="upper left")   
plt.show() 





'''
  labels=['USA','JAPAN','EU'] : x축의 값. 3개의 그룹명
  vert=False : 가로형태의 박스그래프.
  sym 속성  : "r*"  : 이상치 표현기호 및 색상
            r:red. b:blue .....
            *:별, s:정사각형, +:십자가, .:작은점
            o:큰점(기본값), d:다이아몬드            
'''
# 박스그래프: 두개의 그래프 출력하기
usa_df=df[df['origin']=='usa']["mpg"]
japan_df=df[df['origin']=='japan']["mpg"]
eu_df=df[df['origin']=='europe']["mpg"]


data=[usa_df, japan_df, eu_df]
plt.boxplot(data, label=['usa', 'japan', 'europe'], patch_artist=True)
plt.title("제조국자별 연비분포(수직박스플롯)")
plt.show()
data=[usa_df, japan_df, eu_df]
plt.boxplot(data, label=['usa', 'japan', 'europe'], 
            patch_artist=True, vert=False)
plt.title("제조국자별 연비분포(수평박스플롯)")
plt.show()


# 박스그래프: 두개의 그래프 출력하기
fig = plt.figure(figsize=(15,5)) #그래프출력영역,크기지정.
#그래프 출력영역을 분리
ax1 = fig.add_subplot(1,2,1) #1행2열 첫번째 그래프
ax2 = fig.add_subplot(1,2,2) #1행2열 두번째 그래프

ax1.boxplot(data, label=['usa', 'japan', 'europe'], patch_artist=True)
ax2.boxplot(data, label=['usa', 'japan', 'europe'], 
            patch_artist=True, vert=False)

ax1.set_title("제조국자별 연비분포(수직박스플롯)")
ax2.set_title("제조국자별 연비분포(수평박스플롯)")
plt.show()



fig, ax = plt.subplots(1, 2, figsize=(15,5))
ax[0].boxplot(data, label=['usa', 'japan', 'europe'], patch_artist=True)
ax[0].set_title("제조국자별 연비분포(수직박스플롯)")

ax[1].boxplot(data, label=['usa', 'japan', 'europe'], 
            patch_artist=True, vert=False)
ax[1].set_title("제조국자별 연비분포(수평박스플롯)")
plt.tight_layout()
plt.show()

usadf = df[df['origin']=='usa'][['mpg']]
usadf.describe()

'''
              mpg
count  249.000000
mean    20.083534
std      6.402892
min      9.000000
25%     15.000000
50%     18.500000
75%     24.000000
max     39.000000
'''
   

#df데이터를 origin 컬럼으로 그룹화하여 그룹별 합계 출력하기
df.groupby("origin").sum()
#df데이터를 origin 컬럼으로 그룹화하여 그룹별 건수 출력하기
df.groupby("origin").count()
df.origin.value_counts()
#df데이터를 origin 컬럼으로 그룹화하여 그룹별 평균 출력하기
df.groupby("origin").mean()  #error
#df데이터를 origin 컬럼으로 그룹화하여 그룹별 중간값 출력하기
df.groupby("origin").median() #error



#seaborn 모듈 : 시각화모듈, 데이터셋
#               matplot 모듈 확장. 고급시각화.
import seaborn as sns
import matplotlib.pyplot as plt

titanic = sns.load_dataset("titanic")
titanic.info()

'''
  선형회귀그래프(regplot) : 산점도+회귀선 표시
  회귀선 : 모든점에서 가장 가까운 점들을 선으로 표시.
  fit_reg=False : 회귀선 표시 안함
  line_kws={'color': 'red', 'linewidth': 3}
'''

fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(1, 2, 1) 
ax2 = fig.add_subplot(1, 2, 2)
sns.regplot(x='age',  y='fare', data=titanic, ax=ax1)  
sns.regplot(x='age', y='fare',  data=titanic, ax=ax2, fit_reg=False)  
plt.show()


# #히스토그램 작성하기
# fig = plt.figure(figsize=(15, 5))   
# ax1 = fig.add_subplot(1, 3, 1) #1행3열 1번째영역
# ax2 = fig.add_subplot(1, 3, 2) #1행3열 2번째영역
# ax3 = fig.add_subplot(1, 3, 3) #1행3열 3번째영역
# '''
# sns.distplot : 밀도,빈도수 함께 출력.
#                kde=False 지정하면 빈도수만 출력
# sns.kdeplot : 밀도를 출력 그래프               
# sns.histplot : 빈도수 출력 그래프  
# '''
# sns.distplot(titanic['fare'], ax=ax1)
# sns.kdeplot(x='fare',data=titanic, ax=ax2)
# sns.histplot(x='fare',data=titanic,ax=ax3)
# ax1.set_title('titanic fare - distplot')
# ax2.set_title('titanic fare - kdeplot')
# ax3.set_title('titanic fare - histplot')
# plt.show()



# 히트맵 그래프 : 범주형 데이터의 수치를 색상과 값으로 표시 
#pivot_table : 2개의 범주형 데이터를 행열로 분리
#aggfunc='size' : 데이터 갯수
#                 mean, sum...

'''
'sum': 값들의 합계
'mean': 값들의 평균 (기본값)
'count': 그룹 내 항목의 수 (NaN 제외)
'size': 그룹 내 항목의 수 (NaN 포함)
'min', 'max': 최솟값, 최댓값
'std', 'var': 표준편차, 분산
'median': 중앙값
'first', 'last': 그룹의 첫 번째/마지막 값
'nunique': 고유한 값의 수

'''


fare_summary = titanic.pivot_table(index='sex', 
                                   columns='pclass', 
                                   values='fare', aggfunc='mean')

'''
pclass          1          2          3   
sex
female  106.125798  21.970121  16.118810
male     67.226127  19.741782  12.661633
'''



table = titanic.pivot_table\
    (index=['sex'], columns=['class'], aggfunc='size')
table    
'''
class   First  Second  Third
sex                         
female     94      76    144
male      122     108    347
'''


t1=titanic[["sex","class"]].value_counts()
'''
sex     class 
male    Third     347
female  Third     144
male    First     122
        Second    108
female  First      94
        Second     76
Name: count, dtype: int64
'''





'''
  table : 표시할 데이터 
  annot=True : 데이터값 표시 여부
  fmt='d' : 10진 정수로 표시
  linewidth=.5 : 여백, 간격 
  cbar=False : 컬러바 표시 여부
  cmap='YlGnBu' : 컬러맵, Greys
https://matplotlib.org/3.2.1/tutorials/colors/colormaps.html     
'''
sns.heatmap(fare_summary,annot=True,fmt='d',
            cmap='YlGnBu',linewidth=.5,cbar=True)
plt.show()


### boxplot 그래프
fig = plt.figure(figsize=(15, 10))   
ax1 = fig.add_subplot(2, 2, 1) 
ax2 = fig.add_subplot(2, 2, 2) 
ax3 = fig.add_subplot(2, 2, 3) 
ax4 = fig.add_subplot(2, 2, 4) 
'''
data=titanic : 데이터변수명.
x="alive", y="age" : titanic의 컬럼명.

hue='sex' : 성별로 분리(unique()).
violinplot : 값의범주+분포도를 표시. 가로길이 넓은부분은 분포가 많은 수치의미
'''
sns.boxplot(x='alive', y='age', data=titanic, ax=ax1) 
sns.boxplot(x='alive', y='age', hue='sex', data=titanic, ax=ax2) 
sns.violinplot(x='alive', y='age', data=titanic, ax=ax3) 
sns.violinplot(x='alive', y='age', hue='sex', data=titanic,ax=ax4) 
ax2.legend(loc="upper center")
ax4.legend(loc="upper center")
plt.show()


#pairplot : 각각의 컬럼들의 
#           산점도출력. 대각선위치는 히스토그램으로 표시.
#           값의 분포, 컬럼간의 관계.
titanic_pair = titanic[["age","pclass","fare"]]    
titanic_pair
sns.pairplot(titanic_pair)
plt.show()


# FacetGrid : 조건(컬럼의 값)에 따라 그리드 나누기. 
#             컬럼의 값(범주형데이터)에 따라서 여러개의
#             그래프를 출력.
g = sns.FacetGrid(data=titanic, col='who', row='survived') 
g = g.map(plt.hist, 'age') #age 컬럼의 히스토그램 출력
plt.show()





import pandas as pd
chipo = pd.read_csv("data/chipotle.tsv",sep="\t")
chipo.info()

'''
데이터 속성 설명
order_id : 주문번호
quantity : 아이템의 주문수량
item_name : 주문한 아이템의 이름
choice_description : 주문한 아이템의 상세 선택 옵션
item_price : 주문 아이템의 가격 정보
'''


#1. chipo 데이터의 행열의 갯수 출력하기
chipo.shape

#2.컬럼명
chipo.columns

#3. 인덱스명
chipo.index

#4. orderid : str,  item_price : float  타입 변경
'''
#   Column              Non-Null Count  Dtype 
---  ------              --------------  ----- 
 0   order_id            4622 non-null   int64 
 1   quantity            4622 non-null   int64 
 2   item_name           4622 non-null   object
 3   choice_description  3376 non-null   object
 4   item_price          4622 non-null   object
'''

chipo.order_id = chipo.order_id.astype(str)
chipo.info()
type(chipo.item_price)
chipo.item_price = chipo.item_price.str.replace("$","").astype(float)

#5.  상품명을 조회 , 상품의 갯수

chipo.item_name.unique()
len(chipo.item_name.unique())


#6. 주문금액 합계
hap=chipo.item_price.sum()

#7 주문건수
chipo.order_id.unique()
cnt=len(chipo.order_id.unique())

#8 주문당 평균 금액
hap/cnt
g1=chipo.groupby("order_id")
chipo.groupby("order_id")['item_price'].sum().mean()

#9. 50달러 이상 주문한 주문번호를 조회
g2=chipo.groupby("order_id").sum()
result=g2[g2['item_price']>=50]
list(result.index)
result.index

#10. 50달러 이상 주문한 주문정보를 조회

chipo_50 = chipo[chipo.order_id.isin(result.index)]
chipo_50.shape

chipo_51 = chipo.groupby("order_id")\
        .filter(lambda x : sum(x["item_price"]) >=50 )

chipo_51.info()


#11. item_name별 단가 (min)

price_one= chipo.groupby("item_name").min().to_dict()  
#group(min,max...)  함수가 있어야 dictionary로 수정된다 

price_one= chipo.groupby("item_name")

for key, item in price_one:
    print(key, len(item), type(item))


len(chipo["item_name"].unique())

g4=chipo[chipo["item_name"]=='Chicken Bowl']


price_min = chipo.groupby("item_name").min()["item_price"]
price_min.max()

plt.rc("font", family="Malgun Gothic")
plt.hist(price_min, bins=10)
plt.xticks(range(10))
plt.ylabel("상품갯수")
plt.title("상품단가 분포")
plt.show()

price_min.plot(kind = "hist", bins=10)
plt.xticks(range(10))
plt.ylabel("상품갯수")
plt.title("상품단가 분포")
plt.show()


#12. 주문당 금액이 가장 높은 5건의 주문 총수량을 조회하기

chipo.groupby("order_id").sum()\
            .sort_values(by='item_price', ascending=False)[:5]


#13.  Chicken Bowl 몇번 주문되었는지 출력하기
chip_chicken = chipo[chipo['item_name']=='Chicken Bowl'] #729

len(chip_chicken.groupby("order_id"))  #615
len(chip_chicken.drop_duplicates(['item_name', 'order_id'])) #615










###################
#  전세계 음주 데이터 분석하기 : drinks.csv
import pandas as pd
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
beer_wine_corr=\
    drinks[["beer_servings","wine_servings"]].corr()
beer_wine_corr
beer_wine_corr=drinks[["beer_servings","wine_servings"]]\
    .corr(method="pearson")
beer_wine_corr
# 켄달 상관계수 : 샘플 사이즈가 작은 경우.
#               동률데이터의 확율이 높은 경우
beer_wine_corr=drinks[["beer_servings","wine_servings"]]\
    .corr(method="kendall")
beer_wine_corr
# 스피어만 상관계수 : 정규화가 되지 않는 데이터에 많이 사용
beer_wine_corr=drinks[["beer_servings","wine_servings"]]\
    .corr(method="spearman")
beer_wine_corr

drinks.columns
cols = drinks.columns[1:-1]
cols
corr = drinks[cols].corr()
sns.heatmap(corr)
corr
corr.values
#상관계수 시각화하기
#히트맵을 이용하여 시각화 하기



cols_view = ["beer","spirit","wine","alcohol"]
sns.set(font_scale=1.5) #글자크기.
hm=sns.heatmap(corr.values,  #데이터
                cbar=True,   #색상맵
                annot=True,  #데이터값표시
                square=True, #히트맵을 사각형으로 출력
                yticklabels=cols_view, #y축 표시 라벨
                xticklabels=cols_view) #x축 표시 라벨

#seaborn 모듈의 산점도을 이용하여 시각화 하기
sns.pairplot(drinks[cols])
plt.show()

#회귀그래프 작성하기
sns.regplot\
 (x="beer_servings",y="total_litres_of_pure_alcohol",
  data=drinks)
 
#각 변수의 결측값 갯수 조회하기
drinks.info()
drinks.isnull().sum() 
#대륙별 국가수 조회하기
drinks["continent"].value_counts()  #각 value별 합계
drinks["continent"].value_counts(dropna=False)  #null 포함
drinks.groupby("continent").count()["country"]

#continent 컬럼의 결측값을 OT로 변경하기
#fillna : 결측값을 다른 값으로 치환함수
drinks["continent"]=drinks["continent"].fillna("OT")
drinks.info()

import matplotlib.pyplot  as plt

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

cont_mean = drinks.groupby("continent")["total_litres_of_pure_alcohol"].mean()
cont_mean
total_mean = drinks["total_litres_of_pure_alcohol"].mean()
total_mean

drinks.info()


import numpy as np
plt.rc("font",family="Malgun Gothic")

# 2 대륙명 : x축의 라벨
continents=cont_mean.index.tolist()
continents
continents.append("Mean")
x_pos = np.arange(len(continents))

#y축 데이터 : 대륙별 평균값
alcohol = cont_mean.tolist()
alcohol
alcohol.append(total_mean)


#그래프화
#plt.bar : 막대그래프
#bar_list : 막대그래프의 막대 목록
bar_list =plt.bar(x_pos, alcohol, align='center',alpha=0.5)
bar_list
#bar_list[len(continents) - 1] : bar_list[6] 막대
#set_color('r') : 색상 설정. r:red 
bar_list[len(continents) - 1].set_color('r') #7번째 막대색 red로 설정

plt.xticks(x_pos, continents)
plt.ylabel('total_litres_of_pure_alcohol') #y축설명
plt.title('대륙별 평균알콜 섭취랑') #제목
plt.show()

'''
대륙별 beer_servings의 평균를 막대그래프로 시각화
가장 많은 맥주를 소비하는 대륙(EU)의 막대의 색상을 빨강색("r")으로 변경하기 
전체 맥주 소비량 평균을 구해서 막대그래프에 추가
평균선을 출력하기. 평균 막대 색상은 노랑색 ("y")
평균선은 검정색("k--")
'''
# 1 대륙별 맥주 소비량 평균

bear_mean = drinks.groupby("continent").beer_servings.mean()
bear_mean
total_beer = drinks.beer_servings.mean()
total_beer

# 2 대륙명 : x축의 라벨
continents=bear_mean.index.tolist()
continents
continents.append("Mean")
x_pos = np.arange(len(continents))

#y축 데이터 : 대륙별 평균값
alcohol = bear_mean.tolist()
alcohol
alcohol.append(total_beer)


#그래프화
#plt.bar : 막대그래프
#bar_list : 막대그래프의 막대 목록
bar_list =plt.bar(x_pos, alcohol, align='center',alpha=0.5)
bar_list
#bar_list[len(continents) - 1] : bar_list[6] 막대
#set_color('r') : 색상 설정. r:red 
bar_list[len(continents) - 1].set_color('y') #6번 막대색 red로 설정
bar_list[2].set_color('r') #2번 막대색 red로 설정
plt.plot([0,6], [total_beer, total_beer], 'k--')
plt.xticks(x_pos, continents)
plt.ylabel('맥주 소비량') #y축설명
plt.title('대륙별 맥주 섭취랑') #제목
plt.show()

#대한민국은 얼마나 술을 독하게 마시는 나라인가?
#total_servings : 전체 주류 소비량 컬럼 추가
drinks["total_servings"] =\
    drinks["beer_servings"] + \
    drinks["spirit_servings"] +\
    drinks["wine_servings"]

#alcohol_rate : 알콜비율 (알콜섭취량/전체주류소비량) 추가
drinks["alcohol_rate"] = \
    drinks["total_litres_of_pure_alcohol"]/drinks["total_servings"]

drinks.info()
#alcohol_rate 컬럼에 결측값 존재.
#전체주류소비량이 0인 경우 불능 => 결측값
#alcohol_rate 컬럼의 값이 결측값인 레코드 조회하기

#1. alcohol_rate null인 index 구하기
drinks["alcohol_rate"].isnull()
alcoholnull = drinks[drinks["alcohol_rate"].isnull()]
alcoholnull
#2. alcohol_rate 컬럼의 결측값을 0을 치환하기
drinks["alcohol_rate"]=drinks["alcohol_rate"].fillna(0)
drinks.info()


#alcohol_rate의 값으로 내림차순 정렬하기. alcohol_rate_rank 저장
alcohol_rate_rank=drinks.sort_values(by="alcohol_rate", ascending=False)\
                [["country","alcohol_rate"]]

alcohol_rate_rank


alcohol_rate_rank.head()
alcohol_rate_rank.shape

alcohol_rate_rank.country.tolist().index("South Korea")
alcohol_rate_rank.head(15)

#국가명목록
country_list = alcohol_rate_rank.country.tolist()


#1. x축값
x_pos = np.arange(len(country_list))

#2. y축값
rank=alcohol_rate_rank.alcohol_rate.tolist()

#막대그래프 
# bar_list : 막대 목록
bar_list = plt.bar(x_pos, rank)

#대한민국 막대의 색을 red로 변경
korea_rank = country_list.index("South Korea")

bar_list[korea_rank].set_color('r')
plt.ylabel('alcohol rate')
plt.title('liquor drink rank by contry')
plt.axis([0, 200, 0, 0.3]) #xy축범위:[x축값의시작,x축값의종료,y축값의시작,y축값의종료]

#korea_serving_rate : 대한민국 전체 술소비량 데이터. y축값

korea_alcohol_rate = alcohol_rate_rank[ alcohol_rate_rank['country']=='South Korea']\
    ['alcohol_rate'].values[0]
korea_alcohol_rate     #0.0593939393939394


plt.annotate('South Korea : ' + str(korea_rank + 1)+"번째",   # 텍스트
          xy=(korea_rank, korea_alcohol_rate),                # 화살표가 가리키는 지점
          xytext=(korea_rank + 10, korea_alcohol_rate+0.1 ),  # 텍스트  위치할 지점
          arrowprops=dict(facecolor='red', shrink=0.05))      #속성을 정의하는 딕셔너리
            #shrink=0.05는 화살표의 양 끝을 5% 줄인다 
plt.show()

#--------------------------------------------------

# merge

import pandas as pd
import numpy as np
#서울 구별 CCTV 정보 데이터 읽기
CCTV_Seoul = pd.read_csv("data/01. CCTV_in_Seoul.csv")
CCTV_Seoul.info()

#서울 경찰서별 범죄율 정보 데이터 읽기
crime_Seoul = pd.read_csv('data/02. crime_in_Seoul.csv',
                          thousands=',', encoding='cp949')
crime_Seoul.info()
#전국 경찰서 위치 데이터 읽기
police_state = pd.read_csv('data/경찰관서 위치.csv', encoding='cp949')
police_state.info()


#1. 서울지역 경찰서 위치데이터만 저장

police_state.head()
police_Seoul = police_state[police_state["지방청"]=="서울청"]
police_Seoul

#police_Seoul 데이터의 경찰서 컬럼의 내용을 '서울중부'--> '중부서' 이름변경하여
# 관서명 컬럼으로 생성하기
police_Seoul["관서명"]=police_Seoul["경찰서"].str[2:]+"서"
police_Seoul

#2 police_Seoul 데이터에 지방청, 경찰서,구분 컬럼 제거하기
del police_Seoul["지방청"],police_Seoul["경찰서"],police_Seoul["구분"]
police_Seoul.info()


# police_Seoul["관서명"] 중복행을 제거하기
police_Seoul = police_Seoul.drop_duplicates(subset=["관서명"])
police_Seoul
police_Seoul.info()

# police_Seoul 데이터의 주소 컬럼을 이용하여 구별 컬럼을 생성하기
police_Seoul["구별"]=police_Seoul["주소"].str.split().str[1]

# police_Seoul 데이터의 주소 컬럼제거하기
del police_Seoul["주소"]
police_Seoul.info()

# 관서명 컬럼을 연결컬럼으로 crime_Seoul 데이터와 police_Seoul 데이터를 병합하기 
# data_result 데이터에 저장하기
data_result=pd.merge(crime_Seoul,police_Seoul, on="관서명")
data_result.head()

#구별 범죄의 합계를 출력하기
crime_sum =data_result.groupby("구별").sum()
crime_sum 

#범죄 종류(강간,강도,살인,절도,폭력)별 검거율 컬럼 추가하기
# 검거율 = 검거/발생 * 100
col_list = ['강간', '강도', '살인', '절도', '폭력']
for col in col_list:
    crime_sum[col+"검거율"] =\
        crime_sum[col+" 검거"]/crime_sum[col+" 발생"]*100

crime_sum.info()

crime_sum


#검거율 데이터 중 100보다 큰값은 100으로 변경하기
# df.loc(조건문, column명)
# 조건문이 True이면 column명에 값을 넣는다 
for col in col_list:
    crime_sum.loc[crime_sum[col+"검거율"]>=100, col+"검거율"]=100

# df.loc(조건문, column명)
for col in col_list:
    print(crime_sum.loc[crime_sum[col+"검거율"]>100, col+"검거율"])
crime_sum

# 구별 검거율과, CCTV 갯수를 산점도와 회귀선으로 출력하기.
# 오차가 큰 10개 구이름을 그래프로 출력하기
#crime_sum 인덱스를 구별 컬럼으로 변경하기
crime_sum = crime_sum.reset_index()
crime_sum
crime_sum.index
crime_sum.info()
crime_sum

#구별 컬럼으로 CCTV_Seoul,crime_sum 데이터를 병합하여 data_result에 저장하기
CCTV_Seoul.info()

#기관명 컬럼을 구별 컬럼명 변경
CCTV_Seoul.rename(columns={"기관명":"구별"},inplace=True)
CCTV_Seoul.drop\
    (["2013년도 이전","2014년","2015년","2016년"],axis=1,inplace=True)
    
CCTV_Seoul

#cctv데이터+crime 데이터를 구별로 병합    
data_result = pd.merge(CCTV_Seoul,crime_sum, on="구별")
data_result.info()
   
#절도검거율과cctv 회귀선과 산점도 출력하기
#회기선 구하기 (x, y, 차수)
fp1 = np.polyfit(data_result['소계'], data_result['절도검거율'], 2) 
f1=np.poly1d(fp1) #회귀선을 위한 함수
fx=np.linspace(500,4000,100)   



#data_result 데이터에 오차 컬럼을 추가하기
# 실제 검거율과 기대검거율의 차이의 절대값 저장
data_result["오차"]=np.abs(data_result["절도검거율"]-f1(data_result['소계']))
#오차의 내림차순으로 정렬하여 df_sort 데이터 저장
df_sort = data_result.sort_values(by="오차", ascending=False)
df_sort 
    
import matplotlib.pyplot as plt 
plt.rc('font', family ='Malgun Gothic')
plt.figure(figsize=(14,10))

plt.scatter(df_sort['소계'],df_sort["절도검거율"],c=df_sort['오차'], s=50)
plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')    

for n in range(10):
    plt.text(df_sort.iloc[n,]["소계"]*1.001,  #x축값
             df_sort.iloc[n,]['절도검거율']*0.999, #y축값
             df_sort.iloc[n,]['구별'], fontsize=10)
    
plt.xlabel('CCTV 갯수')
plt.ylabel('절도범죄 검거율')
plt.title("CCTV와 절도 검거율 분석")
plt.colorbar()
plt.grid()
plt.show() 
    
####  하위 폴더의 이름을 폴더명과 함께 list에 저장 하세요
###   global file_count, df_data


import os
files=[]
file_count=0 

def read_folder(dirname):
    global file_count, df_data
    folderlist = os.listdir(dirname)
    for f_folder in folderlist:
        if os.path.isdir(dirname+"/"+f_folder):
            read_folder(dirname+"/"+f_folder)
        else:
            file_count +=1  
            print(file_count, dirname+"/"+f_folder) 
            files.append(dirname+"/"+f_folder)



read_folder('C:/Users/letuin/pyworkspace')
print(files)



####  지정 지역의 인구분포와 가장 비슷한 지역찾기
# 1세 ~ 10세
df = pd.read_csv("data/age.csv",encoding="cp949",thousands=",",index_col=0)
col_name=df.columns
df.info()
col_name=col_name[1:13]
df=df[col_name]  # df[column list]
df.info()
df.index

name1= df.index.str.contains("역삼1동") #contains("역삼1동")   
name1  # 한개만 true
df2=df[name1] # df[조건식]
df2.info()
df2.head()

df2 # 지정 지역 데이터  , 역삼1동 row 1개

home=df2.iloc[0][1:]/df2.iloc[0][0]  #역삼1동 비율
# df3 : name1값을 제외한 다른 데이터만 저장


name2 = list(map(lambda x : not x,name1))  # name1의 반대 역삼동 없는 row
name2
df3 = df[name2]
df3.info()


tdf4=df3.T

mn=1
for label, content in df3.T.items():
    #print(content)
    away=content[1:]/content[0]  #  0 ~ 10 / 전체 11개
    s=sum((home-away) ** 2)
    if s < mn :
        mn = s
        result=content[1:]
        name=result.name
        
print(name)
name.find("(")
name=name[:14] # 서울특별시 서대문구 신촌동

plt.figure(figsize = (10,5), dpi=100)      
plt.title('역삼1동 지역과 가장 비슷한 인구 구조를 가진 지역')
plt.plot(df2.iloc[0][1:], label = df2.iloc[0].name)
plt.plot(result, label=result.name)

# serise일 len(result) , value 가 y

plt.xticks(range(len(df2.iloc[0][1:])), col_name[1:],rotation=45,fontsize="small")
plt.legend()
plt.show()   

