
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score






print(left_max,left_min)
    left=(left_max-left_min)/3 
    
    ax1.set_ylim(left_min-left, left_max+left)    


# 차트 폭조정
    right=(right_max-right_min)/4
    print(right_max,right_min, right)
    ax2.set_ylim(right_min, right_max*3) 


url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"

'''
1 - fixed acidity : 주석산농도
2 - volatile acidity : 아세트산농도
3 - citric acid : 구연산농도
4 - residual sugar : 잔류당분농도
5 - chlorides : 염화나트륨농도
6 - free sulfur dioxide : 유리 아황산 농도
7 - total sulfur dioxide : 총 아황산 농도
8 - density : 밀도
9 - pH : ph
10 - sulphates : 황산칼륨 농도
11 - alcohol : 알코올 도수
12 - quality (score between 0 and 10) : 와인등급
'''


###### 예외 처리:예측가능한 오류 발생시 정상처리
# try except 문장
'''
# try except 문장
#다중예외처리 : 하나의 try 구문에 여러개의 except 구문이 존재
#              예외별로 다른 처리 가능
#else 블럭 : 오류 발생이 안된경우 실행되는 블럭
#pass 예약어 : 블럭 내부에 실행될 문장이 없는 경우
'''       




###########################
#  함수와 람다
#함수는 특정 작업을 수행하는 코드 블록이다 
#함수를 사용하면 코드를 재사용할 수 있고, 프로그램을 더 구조적으로 만들 수 있다
#  함수:def 예약어 사용
###########################

'''
  함수 : def 예약어로 함수 정의
         return 값 : 함수를 종료하고 값을 전달
         매개변수 : 함수를 호출할때 필요한 인자값 정의
              가변매개변수(*p) : 매개변수의 갯수를 지정안함. 0개이상. * p 표현
              기본값설정 : (n1=0,n2=0) : 0,1,2개의 매개변수 가능 
              딕션어리 매개변수 (**dic) : 매개변수를 dictionary로 정의후 사용
 
'''





############################
#  Collection : 여러개의 데이터 저장할 수 있는 객체
#   list(리스트) : 배열의 형태.인덱스사용가능. []로 표시함.
#   tuple(튜플) : 상수화된 리스트.변경불가리스트. ()로 표시함
#   set(셋) : 중복불가. 집합  set() , {1,2,3} 로 표시함
#   dictionary(딕셔너리) : 자바의 Map. (key,value)쌍인 객체들 {}로 표시함
############################

'''
  Collection : 데이터의 모임. 
    리스트(list) : 배열. 순서유지. 첨자(인덱스)사용 가능. []
    튜플(tuple) : 상수화된 리스트. 변경불가 리스트.       ()
    딕셔너리(dictionary) : (key,value)쌍 인 객체         {} 
                 items() : (key,value)쌍 인 객체를 리스트 형태로 리턴
                 keys()  : key들만 리스트형태로 리턴
                 values() : value들만 리스트형태로 리턴
    셋(set)  : 중복불가. 순서모름. 첨자(인덱스)사용 불가. 집합표현 객체.  {} 초기화는 set()
                 &, intersection() : 교집합
                 |, union() : 합집합.
                 
  컴프리헨션(comprehension) : 패턴(규칙)이 있는 데이터를 생성하는 방법
  map, filter
'''




ss = '123'
ss = 'Aa123'
ss = 'Aa'
ss = 'AA'
ss = 'aa'
ss = '     '
ss = '  aa   '
ss = '  Aa   '

'''
   print(값) : 화면에 출력하기
   print(값1,값2) : 값을 여러개 출력
   print("{0:d}{1:2d}...".format(값1,값2,...))  형식문자 이용하여 출력
   print("%2d,%3d" % (값1,값2)) : 형식문자 이용하여 값을 여러개 출력
   print(f"{변수1} {변수2}") : 변수에 해당하는 값을 출력
   print(""" 문자열 """) : 여러줄 문자열
   
   문자열 : 문자들의 모임. 인덱스(첨자)를 사용가능
   "문자열"[시작인덱스:종료인덱스+1:증감값]
      시작인덱스 생략시 : 0번부터시작
      종료인덱스 생략시 : 마지막문자까지
      증감값 생략시 : 1씩 증가 
      
   조건문 : if else, if elif else , True if 조건식 else False   

   반복문 : for 변수 in 범위 , while 조건식  
           범위 : range(초기값,종료값+1,증감식)
           break,continue
   조건문,반복문 : 들여쓰기에 주의.         
   
   문자열 함수
     len(문자열) : 문자열의 길이
     문자열.count(문자) : 문자열에서 문자의 갯수 리턴
     문자열.find(문자) : 문자열에서 문자의 위치 리턴  문자가 없는 경우 -1 리턴
     문자열.index(문자) : 문자열에서 문자의 위치 리턴 문자가 없는 경우 오류 발생
     문자열.isdigit() : 숫자?
     문자열.isalpha() : 문자?
     문자열.isalnum() : 숫자또는문자?
     문자열.isupper() : 대문자?
     문자열.islower() : 소문자?
     문자열.isspace() : 공백 ?
'''


