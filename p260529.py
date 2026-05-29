# -*- coding: utf-8 -*-
"""
Created on Fri May 29 09:26:20 2026

@author: user
"""

import pandas as pd
# pip install pandas

import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

# %%
#튜플 데이터를 Series 데이터로
tuple_data = ("홍길동",'1991-01-25','남',True)
sr = pd.Series(tuple_data,index=["이름","생년월일","성별","학생여부"])


print(sr.head())
print(sr.index) #값의 이름
print(sr.values)

sr = pd.Series(tuple_data)
print(sr)
print(sr.index) #값의 이름
print(sr.values)


#리스트를 Series로 만들기
list_data = ["홍길동",'1991-01-25','남',True]
sr = pd.Series(list_data,index=["이름","생년월일","성별","학생여부"])
print(sr)
print(sr.index) #값의 이름
print(sr.values)



#Series 조회

#한개의 값만 조회
print(sr[0])  #순서로 조회.
print(sr["이름"]) #인덱스로 조회
print(sr.이름) #인덱스로 조회A
print(sr[1])  #순서로 조회.
print(sr["생년월일"]) #인덱스로 조회A
print(sr.생년월일) #인덱스로 조회



#여러개의 값 조회 : list, 구간 0:10
print(sr[[0,1]])  #순서로 조회
print(sr[['이름','생년월일']])  #인덱스 조회
#print(sr['이름','생년월일'])  #오류발생
#여러개의 값 조회. 범위 지정하여 조회
print(sr[0:2])  #순서로 조회. 마지막값 앞까지
print(sr['이름':'성별'])  #인덱스 조회. 마지막 값 까지



#=============================딕셔너리 데이터를 Series 데이터로
dict_data = {'a':1,'b':2,'c':3}
sr = pd.Series(dict_data) #Series 객체 생성 
print(sr)
print(sr.index)  #Index(['a', 'b', 'c'], dtype='object')
print(sr.values)  # [1 2 3]


###########  dataframe
#리스트를 이용하여 데이터프레임 객체 생성
df = pd.DataFrame([[15,'남','서울중'],[17,'여','서울여고'],
                   [17,'남','서울고']],
                  index=['홍길동','성춘향','이몽룡'],
                  columns=['나이','성별','학교'])   


###---------------------------

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
df.rename(index={"학생1":"홍길동"})  #inplace=True 효과
print(df)
df = df.rename(index={"학생1":"홍길동"})  #inplace=True 효과
print(df)

# 딕셔너리 이용 (key:columns)
dict_data= {'c0':[1,2,3],'c1':[4,5,6],'c2':[7,8,9],
            'c3':[10,11,12],'c4':[13,14,15]}
df = pd.DataFrame(dict_data)
print(df)
print("컬럼명:",df.columns) #열의이름
print("인덱스명:",df.index) #행의이름 



###------------------------------
exam_data = {"수학":[90,80,70],
             "영어":[98,88,95],
             "음악":[85,95,100],
             "체육":[100,90,90]}

#1
df = pd.DataFrame(exam_data,index=["홍길동","이몽룡","김삿갓"])
print(df)
#2
df = pd.DataFrame(exam_data)
df.index=["홍길동","이몽룡","김삿갓"]
print(df)
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


###---------------------------------
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






###------------------


#과목별 최대점수
print(df.max())
#수학 최대점수
print(df.max()["수학"])
print(df["수학"].max())


print(df.median())
#수학의 중간값
print(df.median()["수학"])
print(df["수학"].median())




df["수학":'영어'] # 에러는 없다

df.loc[:, '수학':'영어']
df.iloc[:, '수학':'영어'] # error
df.iloc[:, 1:2]



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


#수학 통계정보
df.describe()["수학"]
df["수학"].describe()
#간략정보
df.info()
df["수학"].info()  
#데이터의 처음 일부(5개) 조회
df.head()
#데이터의 마지막 5개 조회
df.tail()


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
del df3["음악"]
df3

#copy() : 깊은 복사
df4 = df.copy()
df4
#df4에서 음악,체육 제거하기
del df4["음악"],df4["체육"]
df4


df


exam_data = {"수학":[90,80,70],
             "영어":[98,88,95],
             "음악":[85,95,100],
             "체육":[100,90,90]}

#1
df = pd.DataFrame(exam_data,index=["홍길동","이몽룡","김삿갓"])
print(df)

#1.총점 컬럼 생성하여, 총점의 역순으로 출력하기
df["총점"]=df["수학"]+df["영어"]+df["음악"]+df["체육"]
df['이름']=df.index
df

df=df.reset_index(drop=True)

df.index



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



###############  Pandas 실습

df=pd.read_csv("data/jeju1.csv")
# df1=pd.read_csv("data/jeju1.csv", index_col=0)

# df1.info()


df_excel=pd.read_excel("data/pd_sale_2015.xlsx", 
                          '2015_500', index_col=None)
df_excel.info()
df_excel.columns

# df[col]
df_excel['Customer ID']
# df_excel['Sale Amount'] > 500
df_excel[df_excel['Sale Amount'] > 1000]




df_excel_dic=pd.read_excel("data/pd_sale_2015.xlsx", 
                          sheet_name=None, index_col=None)


df_1 = df_excel_dic['2015_500']
df_2 = df_excel_dic['2015']

df_all=pd.concat([df_1, df_2])

df_low=df_all[df_all['Sale Amount'] < 1000]
df_upp=df_all[df_all['Sale Amount'] >= 1000]



#  파일 저장 
df_low.to_excel('data/df_low.xlsx', index=False)
df_upp.to_excel('data/df_upp.xlsx', index=False)







