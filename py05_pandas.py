# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 12:01:55 2026

@author: letuin
"""

import pandas as pd
# pip install seaborn
import seaborn as sns


tuple_data = ("홍길동",'1991-01-25','남',True)
sr = pd.Series(tuple_data,index=["이름","생년월일","성별","학생여부"])


print(sr.head())
print(sr.index) #값의 이름
print(sr.values)

bsr = pd.Series(tuple_data)
print(bsr)
print(bsr.index) #값의 이름
print(bsr.values)

# 한개의 값만 조회 rangeindex
print(sr[0])  #순서로 조회.  # 
# print(bsr.loc["이름"]) #인덱스로 조회
# print(bsr.이름) #인덱스로 조회A
print(bsr[1])  #순서로 조회.
# print(bsr["생년월일"]) #인덱스로 조회A
# print(bsr.생년월일) #인덱스로 조회


#리스트를 Series로 만들기
list_data = ["홍길동",'1991-01-25','남',True]
lsr1 = pd.Series(list_data,
               index=["이름","생년월일","성별","학생여부"])

list_data = ["홍길동",'1991-01-25','남',True]
lsr2 = pd.Series(list_data)




lsr1[[0,1]]   # error
lsr1[['이름', '생년월일']]


lsr1['이름', '생년월일'] # error
lsr1[0:2]
lsr1['이름': '성별']



#리스트를 이용하여 데이터프레임 객체 생성
df = pd.DataFrame([[15,'남','서울중'],[17,'여','서울여고'],
                   [17,'남','서울고']])
df

df = pd.DataFrame([[15,'남','서울중'],[17,'여','서울여고'],
                   [17,'남','서울고']],
                  index=['홍길동','성춘향','이몽룡'],
                  columns=['나이','성별','학교'])    
print(df)
print(df.index)
print(df.columns)
print(df.head())

#인덱스명 변경하기
df.index=["학생1","학생2","학생3"]
print(df)
#컬럼명 변경하기
df.columns=["age","gender","school"]
print(df)

###----------------------

#rename : 컬럼명,인덱스명의 일부만 변경하기
# inplace=True : 객체 자체 변경
# rename에서 변경 자료는 dictionary임
df.rename(columns={"age":"나이"},inplace=True)   #default    inplace=False
print(df)



#inplace=True 사용하지 않으면, df= 대입구문이 대체 효과.
df.rename(index={"학생1":"홍길동"})  #inplace=False 효과
print(df)
df = df.rename(index={"학생1":"홍길동"})  #inplace=True 효과
print(df)


import pandas as pd
exam_data = {"수학":[90,80,70],
             "영어":[98,88,95],
             "음악":[85,95,100],
             "체육":[100,90,90]}

#1
df = pd.DataFrame(exam_data,index=["홍길동","이몽룡","김삿갓"])


###---------------------

#홍길동 데이터 조회하기
print(df.수학)
print(df["수학"])
#인덱스명으로 조회하기 => 행을 값 조회. .loc 사용
# loc[인덱스명] : 인덱스에 해당하는 행을 조회
# iloc[순서] : 순서 해당하는 행을 조회
print(df.loc["홍길동"]) #홍길동 행(index) 조회
print(df.iloc[0])  #첫번째 행 조회
type(df.loc["홍길동"])  #Series 객체
print(df.loc["홍길동"].mean())  #평균
print(df.loc["홍길동"].median()) #중간값


#df 데이터의 이몽룡(row, index) 학생 점수 조회하기

df.iloc[1] #순서 조회
#df 데이터의 이몽룡,김삿갓 학생 점수 조회하기
df.loc[["이몽룡","김삿갓"]] #인덱스이름
df.iloc[[1,2]] #순서 조회
df.iloc[1:2] #순서 조회
#범위로 조회하기
df.loc["이몽룡":"김삿갓"] #이몽룡부터 김삿갓까지 
df.loc["이몽룡":] #이몽룡부터 끝까지 
df.loc[:"이몽룡"] #처음부터 이몽룡까지 
df.loc[:] #처음부터 끝까지 
df.loc[::2] #처음부터 끝까지 2칸씩 조회
df.loc[::-1] #처음부터 끝까지 역순으로 조회 
df.iloc[1:3] #1번부터 2번까지 
df.iloc[1:] #1번부터 끝까지 
df.iloc[:2] #처음부터 1번까지 
df.iloc[:] #처음부터 끝까지 
df.iloc[::2] #처음부터 끝까지 2칸씩 조회
df.iloc[::-1] #처음부터 끝까지 역순으로 조회 


#mean() : 평균    column기준으로 row(index)의 평균 :  column값의 평균임
print(df.mean())
print(type(df.mean()))
#수학평균
print("수학평균:",df.mean()["수학"])
print("수학평균:",df["수학"].mean()) #Serise.mean() 함수 
print(df["수학"]) #Series 객체
#수학총점
print(df.sum())
print(type(df.sum()))
print(df.sum()["수학"])
print(df["수학"].sum())


#과목별 최대점수
print(df.max())
#수학 최대점수
print(df.max()["수학"])
print(df["수학"].max())
print(df.median())
#수학의 중간값
print(df.median()["수학"])
print(df["수학"].median())

#   데이터의 갯수 홀수 : 가운데 값.
#   데이터의 갯수 짝수 : 가운데 두값의 평균.

# 분산(var()) :(값 - 평균) ** 2  의 합계 / 개수 
# 표준편차(std())는  sqrt(분산) 즉 분산의 제곱근 
# 평균 (+,-) + 1 시그마 : 68%
# 평균 (+,-) + 2 시그마 : 95%
# 평균 (+,-) + 3 시그마 : 99.7%

df.std() #표준편차
df.var() #분산

#기술통계 => 기본적인 수치데이터조회 
df.describe()
type(df.describe())


###--------
#drop() : 행,열 제거하기
#axis=0 : 행을 의미
#axis=1 : 열을 의미
#행 제거하기
df3=df.copy()
df3.drop(["홍길동"],axis=0,inplace=True)
df3
#열 제거하기
df3.drop(["체육"],axis=1,inplace=True)
df3
#열 제거하기
del df3["음악"], df3['영어']
df3

exam_data = {"수학":[90,80,70],
             "영어":[98,88,95],
             "음악":[85,95,100],
             "체육":[100,90,90]}

#1
df = pd.DataFrame(exam_data,index=["홍길동","이몽룡","김삿갓"])


#1.총점 컬럼 생성하여, 총점의 역순으로 출력하기
df["총점"]=df["수학"]+df["영어"]+df["음악"]+df["체육"]
df['이름']=df.index
df.reset_index(drop=True)


#2. 이름 컬럼을 인덱스 설정하기
df.set_index("이름",inplace=True)  #이름 컬럼을 인덱스로 수정
df
#3. 이름의 역순으로 정렬하기
df.sort_index(ascending=False,inplace=True)
df
#4. 과목별 역순으로 정렬하기
df.info()
df.sort_values(by="영어",ascending=False,inplace=True)
df
df.sort_values(by="음악",ascending=False,inplace=True)
df

df.sort_values(by="총점",ascending=False,inplace=True)
df
#5. 조건에의 한 분류 df[조건식식]
df[df["총점"] >= 355]

#  Ex 

df=pd.read_csv("data/jeju1.csv")


# https://github.com/mskim3578/lesson

# 1) jeju1.csv download
# 2) workspace 에 'data' folder를 만든다

# 1.  간략 정보
df.info()

# 2. 상위 10개 조회
df.head(10)

# 3. 하위 10개 조회
df.tail(10)

# 4. 장소만 조회
df['장소']
df.장소

# 5. LON, LAT 한번에 조회 하기
df[['LON', 'LAT']]

# 6 장소가 돔베돈 인 위도 경동를 프린트 해줘
# df[조건식]  : row 수집

df[df['장소']=='돔베돈']

# 7. 장소 컬럼을 인덱스로 변경 하기
df=df.set_index('장소')
df.info()
# 8. 돈베돈의 경도 위도 조회하기
df.loc['돔베돈']

# 9. 인덱스를 '여행지' 컬럼에 추가 하기
df['여행지']=df.index

# 10 여행지 커럼을 삭제 하세요
del df['여행지']
df


df.to_csv('data/t1.csv')
df.to_csv('data/t2.csv', index=False)

df.to_csv('data_folder/t1.csv')
df.to_csv('data_folder/t2.csv', index=False)


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
#seaborn 모듈에 저장된 데이터셋 목록
sns.get_dataset_names()

#titanic데이터 로드. 
titanic = sns.load_dataset("titanic")
titanic.info()


#1. pclass,class 데이터만 조회하기
titanic[["pclass","class"]]

#2. 컬럼별 건수 조회하기
titanic.count()

#3. 건수 중 가장 작은 값 조회하기
titanic.count().min()

#4. 건수 중 가장 작은 값의 인덱스 조회하기
titanic.count().idxmin()

#5. titanic 데이터의 나이 조회. 최대 나이를 가진 5개의 나이 조회

titanic.sort_values(by='age', ascending=False)['age'].head()

# 6 titanic 데이터의 나이별 인원수를 조회. 최대 인원수를 가진 5개의 나이 조회

titanic['age'].value_counts().head()

#7. titanic 데이터의 최대나이와 최소나이 조회
titanic['age'].max()
titanic['age'].min()

#8. 승객 중 최고령자의 정보 조회하기
titanic['age'].max()
titanic['age'].idxmax()
titanic.iloc[titanic['age'].idxmax()]

titanic[titanic['age']==titanic['age'].max()]

#9 데이터에서 생존건수(342), 사망건수(549) 조회하기
titanic['survived'].value_counts()
titanic['alive'].value_counts()


# 10 성별로 생존건수 조회하기
cntdf=titanic[["sex","survived"]].value_counts()
cntdf.index

cntdf.loc[('female', 1)]

# 11 상관 계수 : -1 ~ 1사이의 값. 변수의 상관관계 수치로 표현
titanic[['pclass','survived']].corr()
titanic['pclass']
titanic[["survived","age", "fare"]].corr()
titanic.info()


######   mpg자료
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

#1.  제조국별 자동차 건수 조회하기
mpg['origin'].value_counts()

#2. 제조국 컬럼의 값의 종류를 조회하기. 
# unique() : 중복을 제거하여 조회 
mpg['origin'].unique()

#3. # 출시년도의 데이터 조회하기
mpg['model_year'].value_counts()
mpg['model_year'].unique()
#4. mpg 데이터의 통계정보 조회하기
mpg.describe()

#5. #mpg데이터의 행의값,열의값 조회
mpg.shape

#6. 모든 컬럼의 자료형을 조회하기
mpg['mpg'].dtypes

