# -*- coding: utf-8 -*-
"""
Created on Thu May 28 09:28:04 2026

@author: user
"""

#  map 함수 
#  map(function, collection)

def square(x):
    return x * x

numbers = [1,2,3,4,5]

# map 사용
result1 = map(square, numbers)
result2=[map(square, numbers)]

print(result1)
print(result2)

print(list(result1)) #[1, 4, 9, 16, 25]
print(list(result2)) #[<map object at 0x000001C8FE599C30>]


[x*x for x in numbers]


###----------------int 함수 이용
before = ["2024","11","08"]
print(type(before[0])) # 문자열 <class 'str'>
print(before)  # ['2024', '11', '08']
#before의 요소 각각을 int 형변환한 후 다시 list 객체로 생성 
after = list(map(int,before))
print(type(after[0])) #<class 'int'>
print(after)   #[2024, 11, 8]

[int(x) for x in before]


# 일반 함수

dataList = [x for x in range(1,11)]

def evenNumber(dataList):
    resList=[]
    for i in dataList:
        if i % 2 ==0:
            resList.append(i)
    return resList

print(evenNumber(dataList))

#### filter
def evenNum(x):
    return x % 2 == 0

evenNum(5)
evenNum(6)

print(list(filter(evenNum, dataList)))





###------------함수 
#함수는 특정 작업을 수행하는 코드 블록이다 
#함수를 사용하면 코드를 재사용할 수 있고, 프로그램을 더 구조적으로 만들 수 있다


def func1() :
    print("func1() 함수 호출됨")
    return 10  #함수 종료 후 값을 리턴

def func2(num) :
    print("func2() 함수 호출됨:",num)
     #리턴값이 없는 함수
a=func1()      # func1() 함수 호출됨
print(a)       # 10
b=func2(100)   # func2() 함수 호출됨: 100
print(b)       #   None
func2('abc')   # func2() 함수 호출됨: abc



def aa():
    print('a')


aa()


###-------지역변수, 전역변수
def func3() :
    c=300 #지역변수
    print("func3() 함수 호출:",a,b,c)
def func4() :
    a=110 #지역변수
    b=220 #지역변수
    print("func4() 함수 호출:",a,b)
#함수 내부에서 전역 변수값 수정하기
def func5() :
    global a,b   #a,b 변수는 전역변수를 사용함
    a=110
    b=220
    print("func5() 함수 호출:",a,b)
    
# a=10
# b=20
func3()
print(a,b,c)


func4()
func5()

#매개변수
def add1(v1,v2):
    return v1+v2
def sub1(v1,v2):
    return v1-v2

hap = add1(10,20)   # 30
sub = sub1(10,20)     # -10
print(hap)
print(sub)

hap = add1(10.5,20.1)
sub = sub1(10.5,20.1)
print(hap)          # 30.6
print(sub)          # -9.600000000000001



###--------
#가변 매개 변수 : 매개변수의 갯수를 정하지 않음 경우
def multiparam(* p) :
    print(p)
    result = 0
    for i in p :
        result += i
    return result


print(multiparam())    # 0
print(multiparam(10))  # 10
print(multiparam(10,20))    # 30
print(multiparam(10,20,30)) # 60
print(multiparam(1.5,2.5,3)) #7.0
print(multiparam("1.5",2.5,3))  #result += i error


###---
# 딕셔너리 가변인자 **kwargs

def dictDefine(**kwargs):
  
    print('='*30)
    print(kwargs)
#     kwargs.sort() 에러
    for k in kwargs:
        print(k,':', kwargs[k])
    print('\n딕셔너리의 총 길이는?', len(kwargs))
# 함수 호출
dictDefine()
dictDefine(a='apple', b='banan', c='carrot')
dictDefine(n='nano', u='umbrella', m='moutain', s='sweet', d='dress')




###---
#매개변수에 기본값 설정
def hap1(num1=0,num2=1) :  #매개변수가 없는 경우 0,1 기본값 설정됨
    return num1+num2

print(hap1())    #num1=0,num2=1 기본값 설정
print(hap1(10))  #num1=10,num2=1 기본값 설정
print(hap1(10,20)) #num1=10,num2=20
print(hap1(0,20))  #num1=0,num2=20


###---
# 람다식을 이용한 함수
hap2=lambda num1,num2:num1+num2  
#print(hap2(10))  #오류 
print(hap2(10,20))  #30
print(hap2(10.5,20.5)) #31.0

#기본값 매개변수
hap3=lambda num1=0,num2=1:num1+num2
print(hap3(10))  #11
print(hap3(10,20))  #30
print(hap3(10.5,20.5)) #31.0


###-----
mylist = [1,2,3,4,5]
add = lambda num:num+10
mylist = list(map(add,mylist))
print(mylist)    #[11, 12, 13, 14, 15]
#num : mylist의 요소값 한개.
mylist = list(map(lambda num:num-10,mylist))
print(mylist) #[1,2,3,4,5]
mylist = list(map(lambda num:num*10,mylist))
print(mylist)  # [10, 20, 30, 40, 50]


###----
# 복수 리스트에 적용하는 람다함수

list1=[1,2,3,4]  
list2=[10,20,30,40]
list3=[10, 20, 30, 40, 50]
hap=lambda n1,n2:n1+n2
haplist=list(map(hap,list1,list2))
print(haplist)   #[11, 22, 33, 44]
# haplist = mylist + list1 + list2
# 리스트 중 최소요소갯수의 리스트의 갯수로 맞춤.
list1.append(0)
list2.append(0)
haplist = list(map(lambda n1,n2,n3:n1+n2+n3,list1,list2, list3))
print(haplist)   #[21, 42, 63, 84, 50] 


###----
    
#mystr 문자열에 파이썬 문자의 위치를 strpos 리스트에 저장하기
mystr = "파이썬 공부 중입니다. 파이썬을 열심히 공부합시다"

#1 find를 이용한 문자 찾기
strpos = []
index=0
while True :
    index = mystr.index("파이썬",index) #index 이후부터 검색 
    if index < 0: #없다
        break
    strpos.append(index)  #0,13
    index += 1            #14
print(strpos)      #[0, 13]



#2 index 함수 사용. 예외처리
strpos = []
index=0
while True :
  try :
    index = mystr.index("파이썬",index)
    strpos.append(index)  #0,13
    index += 1            #14
  except :  #오류 발생시 호출되는 영역
    break  
print(strpos)    

#다중예외처리 : 하나의 try 구문에 여러개의 except 구문이 존재
#              예외별로 다른 처리 가능
num1 = 1
num2 = "A"
num2 = 0
# n1 = int(num1)
# n2 = int(num2)
# print(n1/n2)
try :
    n1 = int(num1)
    n2 = int(num2)
    print(n1+n2)
    print(n1/n2)
except ValueError as e:
    print("숫자로 변환 불가")
    print(e)
except ZeroDivisionError as e :
    print("두번째 숫자는 0안됨")
    print(e)
finally :  #정상,예외 모두 실행되는 구문
    print("프로그램 종료")   


# 다중 예외를 하나로 묶기
num1 = 1
num2 = "A"
try :
    n1 = int(num1)
    n2 = int(num2)
    print(n1+n2)
    print(n1/n2)
except (ValueError, ZeroDivisionError)  as e:
    print("입력 오류")
    print(e)

finally :  #정상,예외 모두 실행되는 구문
    print("프로그램 종료")   


        
# raise : 예외 강제 발생
try :
  print(1)    
  raise ValueError
  print(2)
except ValueError :
  print("ValueError 강제 발생")  
  
#pass 예약어 : 블럭 내부에 실행될 문장이 없는 경우
n=9
if n>10 :
    pass
else :
   print("n의 값은 10 이하입니다.")    
    
try :
    age = int(input("나이를 입력하세요"))
    if age < 19:
        print("미성년")
    else :
        print("성년")  
except ValueError :
    pass    #오류 발생시 무시.   

def dumy() :
    pass


#파일 쓰기 :콘솔에 내용을 입력 받아 파일로  저장 하기
#현재 폴더에 data/data.txt에 저장한다

outfp = open ("data/data.txt", "w", encoding="UTF-8")
while True :
    outstr = input("내용입력 =>")
    if outstr == '':
       break
        
    outfp.writelines(outstr+"\n")
outfp.close()   #!!! close를 해야 저장을 확인 할수 있다 

# data.txt 파일을 읽어서 화면에 출력하기
infp = open("data/data.txt","r",encoding="UTF-8")

while True :
    instr = infp.readline() #한줄씩 읽기
    if instr == None or instr == "" :
        break
    print(instr,end="")
infp.close()    



#이미지 파일 읽어 복사하기
#apple.gif 파일을 읽어서 apple2.gif로 복사하기

infp = open("data/apple.gif", "rb" )
outfp = open("data/apple2.gif", "wb")

while True:
    indata = infp.read()
    if not indata : #파일의 끝 EOF(End of File)
       break 
    outfp.write(indata)
infp.close()
outfp.close()


'''
파일 : open(파일명,모드,[encoding])
         os.getcwd() : 작업폴더 조회
         os.chdir()  : 작업폴더 변경
         os.path.isfile(file) : 파일?
         os.path.isdir(file)  : 폴더?
         os.listdir()  : 폴더의 하위 파일 목록 조회     


'''    
   
import os
#현재 작업 폴더 위치 조회
print(os.getcwd())    
#작업 폴더의 위치 변경
os.chdir("c:/Users/user")
os.chdir("c:/Users/user/pyworkspace")
print(os.getcwd())   
print(os.listdir())

for ss in os.listdir():
    print(ss)


file = os.getcwd()
file

if os.path.isfile(file) :
    print(file,"은 파일입니다.")
elif os.path.isdir(file) :    
    print(file,"은 폴더입니다.")

if os.path.exists(file) :    
    print(file,"은 존재합니다.")
else :    
    print(file,"은 없습니다.")


len(os.listdir())   # 폴더 아래에 있는 파일을 (폴더 포함) 보여준다
#현재 작업폴더
cwd = os.getcwd();
cwd
for f in os.listdir() :
    if os.path.isfile(f) :
        print(f,":파일, 크기:",os.path.getsize(f))
    elif os.path.isdir(f) :
        os.chdir(f)
        print(f,":폴더, 하위파일의갯수:",len(os.listdir()))
        os.chdir(cwd)
        
os.mkdir('temp')       
os.rmdir('aaa')  # 파일 안에 내용이 있을때는 불가함

