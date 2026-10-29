# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 16:44:46 2026

@author: letuin
"""
import numpy as np
np.random.rand()
#배열 생성
#np.arange(15) : 0 ~ 14까지의 숫자를 1차원 배열로 생성
#reshape(3,5) : 3행5열의 2차원배열로 생성.
#               배열 갯수가 맞아야 함.
x= np.arange(15)
a = x.reshape(3,5)
a  #0~14까지의 숫자를 3행 5열의 2차원배열로 생성
type(a)


x=np.arange(0, 10, 2)

# %%
#배열 요소의 자료형
a.dtype  #int32 => 32비트, 4바이트
#배열 형태
a.shape #(3,5) : 3행 5열 2차원 배열

np.arange(15).shape #(15,) 1차원배열
np.arange(15).reshape(15,1).shape #(15, 1) 2차원배열

#배열의 차수
a.ndim  #2차원
x.ndim 
np.arange(15).ndim #1
#배열의 요소의 바이트 크기
a.itemsize  #4 byte
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

###########  난수
# 난수를 이용하여 0~9사이의 정수값을 가진 임의의수를 3행4열
# 배열 생성
#np.floor: 작은 근사 정수
#np.ceil : 큰 근사 정수
np.random.random((3,4))  #0 ~ 1
np.random.random((3,4)) * 10
h=np.floor(np.random.random((3,4)) * 10)
h
h.ndim
h.shape


#h배열을 1차원배열 h1 변경하기
h1=h.ravel() #h배열이 변경되지 않음
h1.ndim
h1.shape


#---------------------

#난수를 이용한 배열 생성
rg = np.random.default_rng(1) #seed값 설정
rg
a = rg.random((2,3)) #2행3열배열. 
a
a = rg.random((2,3)) #2행3열배열. 
a

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
k2=np.hsplit(k,3) #3개로 열 list로 분리 
k3=np.vsplit(k,2) #2개로 행 list로 분리 
k.shape
#k배열의 모든 요소값을 100으로 변경하기
#k=100. k변수에 100 정수값을 저장. k값은 배열이 아님.
k.shape
k[0,0]=100
k[0,:]=100
k[:]=100
k[:,:]=200
k



#+++++++
#2개의 배열을 합하기
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 수직으로 쌓기
vresult = np.vstack((a, b)) #행기준 합. 열의 갯수가 같아야 함
hresult = np.hstack((a, b)) #열기준 합. 행의 갯수가 같아야 함.




#+++++++


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
r=r.sort()
r





#+++++++++++
#0~3사이의 수를 중복없이 5개 선택.
#오류. 중복되어야 함
s=np.random.choice(4,5,replace=False)  #오류. 불가능선택
s=np.random.choice(4,5,replace=True)
s

#+++++++++++



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

