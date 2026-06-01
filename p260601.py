# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 09:21:57 2026

@author: user
"""



##################################################
# titanic 데이터셋 연습(데이터 전처리)
# seaborn 모듈에 저장된 데이터
'''
survived	생존여부
pclass	좌석등급 (숫자)
sex	성별 (male, female)
age	나이
sibsp	형제자매 + 배우자 인원수
parch: 	부모 + 자식 인원수
fare: 	요금
embarked	탑승 항구
class	좌석등급 (영문)
who	성별 (man, woman)
adult_male 성인남자여부 
deck	선실 고유 번호 가장 앞자리 알파벳
embark_town	탑승 항구 (영문)
alive	생존여부 (영문)
alone	혼자인지 여부
'''

#seaborn 모듈 : 시각화모듈, 데이터셋
import pandas as pd
import seaborn as sns #시각화모듈
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

# %%


























#seaborn 모듈에 저장된 데이터
sns.get_dataset_names()

titanic = sns.load_dataset("titanic")
titanic.info()

# 1. 타이타닉의 자료를 10개 조회
titanic.head(10)

# 2.pclass, class  10개  데이터 조회
titanic[['pclass', 'class']].head(10)

# 3. 컬럼별 count 조회하기
titanic.count()

# 4. 건수중에 가장 작은 값을 조회
titanic.count().min()

# 5. 건수중에 가장 작은 값의 인덱스를 구하시오
titanic.count().idxmin()


# 6. titanic의 age,fare 컬럼만을 tidf 데이터셋에 저장하기
tidf = titanic[['age', 'fare']]



# 7. tidf의 평균 데이터 조회 하기
tidf.mean()
# titanic.mean()  # error 숫자 아닌것 이 존재

# 8. tidf 데이터의 나이 조회. 최대 나이를 가진 5개의 나이 조회
tidf.sort_values(by='age', ascending=False).head()

# 9. tidf 데이터의 인원수가 많은 나이 10개 조회 
tidf.age.unique()
tidf.age.value_counts().head(10)


# 10. tidf 데이터의 최대나이와 최소나이 조회

tidf.age.max()
tidf.age.min()

# 11. 승객 중 최고령자의 정보 조회하기

tidf.age.idxmax()
titanic.iloc[630]

# 12. 데이터에서 생존건수(342), 사망건수(549) 조회하기
titanic['survived'].value_counts()
titanic['alive'].value_counts()
titanic[['survived','alive']].value_counts()


h1=titanic[['survived', 'age', 'fare']].corr()


sns.heatmap(h1)


#seaborn 데이터에서 mpg 데이터 로드하기
'''
mpg : 연비
cylinders : 실린더 수
displacement : 배기량
horsepower : 출력
weight : 차량무게
acceleration : 가속능력
model_year : 출시년도
origin : 제조국
name : 모델명
'''
mpg = sns.load_dataset("mpg")
mpg.info()


# 1. 제조국별 자동차 건수 조회하기
mpg.origin.value_counts()
mpg['origin'].value_counts()

# 2. 제조국 컬럼의 값의 종류를 조회하기. 
mpg.origin.unique()

# 3. 출시년도의 데이터 조회하기
mpg.model_year
mpg.model_year.unique()
mpg.model_year.value_counts()
# 4. mpg 데이터의 통계정보 조회하기
mpg.describe()

# mpg 데이터의 행의값 열의 값
mpg.shape  #(398, 9)
mpg.shape[0] # 398
mpg.shape[1] # 9

# 데이터의 모든 컬럼의 자료형

mpg.dtypes

# 1. mpg 데이터의 mpg, weight 컬럼의 최대값을 조회 하기
mpg.mpg.max()
mpg.weight.max()
# 2. mpg 데이터의 mpg, weight 컬럼의 기술 통계 값을 조회 하기
mpg[['mpg','weight']].describe()

# 3. 최대 연비를 가진 자동차의 정보를 조회 하기
mpg.iloc[mpg.mpg.idxmax()]



# 4. mpg 컬럼간의 상관계수 조회 하기
mpg.info()
mpg[mpg.columns[:-2]].corr()


## 결측치 처리 
titanic = sns.load_dataset("titanic")
titanic.info()

#1.  deck의 종류
titanic.deck.unique()
#2.  deck 컬럼의 값별 건수 출력하기
titanic.deck.value_counts()
#3.  결측값을 포함한 값의 건수
titanic.deck.value_counts(dropna=False)

# 4. isnull() 
titanic.deck.isnull()

# 5. notnull() 
titanic.deck.notnull()

# 6  deck가 not null 자료를 t_notnull으로 저장 
####### df[조건식]
t_notnull=titanic[titanic.deck.notnull()]
t_null=titanic[titanic.deck.isnull()]


# 결측값의 갯수
titanic.isnull().sum()
titanic.isnull().sum(axis=0)
titanic.isnull().sum(axis=1)

# 결측값이 아닌 갯수
titanic.notnull().sum()
titanic.notnull().sum(axis=0)
titanic.notnull().sum(axis=1)


########################
#dropna : 결측값 제거 
#         inplace=True 있어야 자체 변경 가능


t1=titanic.dropna()
#  치환하기 fillna
#1. age 컬럼의 값이 결측값인 경우 평균 나이로 변경하기
age_mean= titanic.age.mean()
titanic['age'].fillna(age_mean, inplace=True)

titanic.info()

#2. embark_town 컬럼의 결측값은 빈도수가 가장 많은 
#   데이터로 치환하기
# embark_town 중 가장 건수가 많은 값을 조회하기
#value_counts() 함수 결과의 첫번째 인덱스값.-가장 큰수
embark_town=titanic['embark_town']

t2=embark_town.value_counts()

most_freq=t2.index.to_list()[0]
most_freq=t2.index[0]

titanic['embark_town'].fillna(most_freq, inplace=True)

titanic.info()

# method="ffill" : 앞의 데이터로 치환
# method="bfill" : 뒤의 데이터로 치환
# method="backfill" : 뒤의 데이터로 치환


# titanic['embarked'] null 확인

# titanic['embarked'].isnull() 인 자료를 조회
titanic['embarked']
# df[조건식]
titanic[titanic['embarked'].isnull()]

titanic['embarked'].fillna(method="backfill", inplace=True)
titanic['embarked'][58:65]
titanic['embarked'][825:831]
titanic.info()

#중복데이터 처리
df = pd.DataFrame({"c1":['a','a','b','a','b'],
                   "c2":[1,1,1,2,2],
                   "c3":[1,1,2,2,2]})
df
#duplicated() : 중복데이터 찾기. 
#            중복된 경우 중복된 두번째 데이터 부터 True리턴  
#            전체 행을 비교하여 중복 검색


df_dup = df.duplicated()

df[df_dup]

# 중복자료 제거
df2=df.drop_duplicates()


mpg=sns.load_dataset('mpg')
# value 수정
mpg['kpl']=mpg['mpg']*0.425
mpg['kpl']=mpg['kpl'].round(1)
mpg.info()

### 타입 변환
# 숫자 컬럼에 문자 한개 있을때

import numpy as np

df2 = pd.DataFrame({"c1":['a','a','b','a','b'],
                   "c2":[1,1,'?',2,2],
                   "c3":[1,1,2,2,2]})
df2.info()


df2.replace("?",np.nan,inplace=True)
df2.info()


#범주형 : category형
mpg=sns.load_dataset('mpg')
mpg["origin"].unique()
mpg["origin"].value_counts()
mpg.info()
mpg["origin"]=mpg["origin"].astype('category')
mpg["origin"]=mpg["origin"].astype('str')


# 날자데이터
# 20220101 부터  이후 6까지일 날짜를 데이터 
# date_range : 날짜의 범위를 지정
# 단위 설정   대소문자 구분 않함
#  freq="D" : 일자기준. 기본값
#  freq="M" : 월의 종료일 기준
#  freq="MS" : 월의 시작일 기준
#  freq="3M" : 3개월의 종료일 기준


dates = pd.date_range('20260105', periods=6, freq="d")
dates
dates = pd.date_range('20260105', periods=6, freq="m")
dates
dates = pd.date_range('20260105', periods=6, freq="MS")
dates
dates = pd.date_range('20260105', periods=6, freq="3M")
dates


#주식 데이터 읽기
stock = pd.read_csv("data/stock-data.csv")
stock.info()
stock

stock['new_date']=pd.to_datetime(df['Date'])


# new_date 컬럼에서 년, 월, 일 정보를 가진 컬럼을 추가 한다
stock['Year']=stock["new_date"].dt.year
stock['Month']=stock["new_date"].dt.month
stock['Day']=stock["new_date"].dt.day


##################
#  groupby 함수 : 컬럼으로 데이터 분리. 

titanic = sns.load_dataset("titanic")

#class 컬럼으로 데이터 분할하기
#class 컬럼의 값으로 데이터를 분리 저장
titanic["class"].value_counts()
titanic["class"].unique()   # ['First', 'Second', 'Third']


grouped = titanic.groupby('class')

group_f=grouped.get_group('First')

# 
for key, group in grouped:
    print('key:',key, end=",")
    print("cnt:", len(group), type(group))



############  numpy


#배열 생성
#np.arange(15) : 0 ~ 14까지의 숫자를 1차원 배열로 생성
#reshape(3,5) : 3행5열의 2차원배열로 생성.
#               배열 갯수가 맞아야 함.
x= np.arange(15)
a = x.reshape(3,5)
a  #0~14까지의 숫자를 3행 5열의 2차원배열로 생성
type(a)



#배열 요소의 자료형
a.dtype  #int64 => 32비트, 4바이트
#배열 형태
a.shape #(3,5) : 3행 5열 2차원 배열

np.arange(15).shape #(15,) 1차원배열
np.arange(15).reshape(15,1).shape #(15, 1) 2차원배열

#배열의 차수
a.ndim  #2차원
x.ndim 
np.arange(15).ndim #1
#배열의 요소의 바이트 크기
a.itemsize  #8 byte
#배열의 요소의 갯수
a.size 
np.arange(15).size


#리스트로 2차원 배열 생성하기
d=np.array([[6,7,8],[1,2,3]])
d
type(d)
#0으로 초기화된 3행 4열 배열 e 생성하기
e=np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0]])
e
e.shape

#zero값  setting
e=np.zeros((3,4))
e
e.shape

# 모든 요소의 값이 0인 배열 100개를 1차원으로 생성하기
f = np.zeros(100)
f.shape

#1 값  setting
# 모든 요소의 값이 1인 배열 100개를 1차원으로 생성하기
g = np.ones(100)
g.shape
#1으로 초기화된 10행 10열 배열 h생성하기
h =np.ones((10,10))
h
h.shape
np.eye(10,10) #단위 행렬


#0~9999까지의 값을 가진 배열을 100행 100열 배열 i생성하기
i = np.arange(10000).reshape(100,100)
i.shape
i

#0~2까지의 숫자를 9개로 균등분할하여 배열 생성
j=np.linspace(0,2,9)
j
j.size
#0~9까지의 숫자를 20개로 균등분할한 배열 생성
k=np.linspace(0,9,20)
k
k.size
#정수형 1의 값으로 10개를 가진 배열 l
l = np.ones(10,dtype=int)
l
l.dtype

#상수값 
np.pi

# numpy 데이터 연산
#1차원 배열의 연산
a = np.array([20,30,40,50])
b = np.arange(4) #(0,1,2,3)
c = a-b  #각각의 요소들을 연산
c # array([20, 29, 38, 47])

c = a+b  #각각의 요소들을 연산
c # array([20, 31, 42, 53])

c = a*b  #각각의 요소들을 연산
c # array([ 0,  30,  80, 150])


c = b**2 #b 요소들 각각의 제곱
c

c = a < 35 # a배열의 요소을 35와 비교하여 작으면 True,크면 False
c

#0부터11까지의 숫자를 3행4열 2차원 배열 d로 생성하기
d=np.arange(12).reshape(3,4)
d
#1행1열의 값을 조회하기
d[1,1]
d[0:2,0:2] #1행까지, 1열까지 조회
d[:2,:2] #1행까지, 1열까지 조회
d[::2,::2] #2씩증가. 


#1값으로 채워진 10행 10열 배열 e 생성하기
e=np.ones((10,10))
e
#e배열의 가장 자리는 1로 내부는 0으로 채워진 배열로수정하기
e[1:9,1:9]=0
e
e=np.ones((10,10))
e[1:-1,1:-1]=0
e

#e배열과 같은 모양의배열 f 생성하기
f = np.zeros((8,8))
f

f=np.pad(f,pad_width=1,constant_values = 1)
f
f.shape


#난수를 이용하여 0~9사이의 정수값을 가진 임의의수를 3행4열
# 배열 생성
#np.floor: 작은 근사 정수
#np.ceil : 큰 근사 정수
np.random.random((3,4))
np.random.random((3,4)) * 10
h=np.floor(np.random.random((3,4)) * 10)
h
h.ndim
h.shape

#h배열을 1차원배열 h1 변경하기
h1=h.ravel() #h배열이 변경되지 않음
h1.ndim
h1.shape

rg=np.random.default_rng(1)
rg.random((2,3))

'''
Out[287]: 
array([[0.51182162, 0.9504637 , 0.14415961],
       [0.94864945, 0.31183145, 0.42332645]])

rg.random((2,3))
Out[288]: 
array([[0.82770259, 0.40919914, 0.54959369],
       [0.02755911, 0.75351311, 0.53814331]])

rg.random((2,3))
Out[289]: 
array([[0.32973172, 0.7884287 , 0.30319483],
       [0.45349789, 0.1340417 , 0.40311299]])

'''

#0~9사이의 정수형 난수값을 가진 2행2열 배열 생성
#randint: 정수형 난수 리턴. 
i=np.random.randint(10,size=(2,2))
i
j=np.random.randint(10,size=(2,2))
j
import random
random.seed(43)
i=random.randint(1,100)
i
j=random.randint(1,100)
j

#배열 나누기
k = np.random.randint(10,size=(2,12))
k
np.hsplit(k,3) #3개로 열을 분리. 
np.vsplit(k,2) #2개로 행을 분리. 
k.shape
#k배열의 모든 요소값을 100으로 변경하기
#k=100. k변수에 100 정수값을 저장. k값은 배열이 아님.
k.shape
k[0,0]=100
k[0,:]=100
k[:]=100
k[:,:]=200
k

#choice 함수 : 값을 선택.
#    choice(값의범위,선택갯수,재선택여부)
#    choice(값의범위,선택갯수,확률)
#(10,5,replace=False)
# 10 : 0~ 9사이의 값
# 5 : 5개 선택
# replace=True|False : 중복가능|중복불가
q=np.random.choice(10,5,replace=True)
q

#1~45사이의 수를 중복없이 6개를 선택한 r배열 생성
r = np.random.choice(45,6,replace=False) + 1
r
#정렬
r.sort()
r
# array([14, 33, 34, 36, 38, 42],



#확률 적용 선택
#choice(값의범위,선택갯수,확률)
fruits = ["apple","banana","cherries","durian","grapes"]
u=np.random.choice(fruits,100,p=[0.1,0.2,0.3,0.2,0.2])
u
listu = list(u)
for d in fruits :
    print(d,"=",listu.count(d))
  
# p=[0.1,0.2,0.3,0.2,0.1,0.1] 
# p의 전체 합: 1 
p=[0.1,0.2,0.3,0.2,0.1,0.1] 
sum(p)

q=np.random.choice(4,5,replace=False)
q













