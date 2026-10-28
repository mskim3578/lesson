# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:40:23 2026

@author: Admin
"""

##### 02 python 기본 문법
###02-01 프로그램밍 기본  

  
# 문장의 연결
# ctl + 1

# 실행 F9
# %%
# shift + enter
text1 = "안녕 하세요 \
       렛유인 입니다 "
print(text1)

#print(text1)
'''
fdfd
fdfdfdfd
fdfdfd
'''

"""
fdfkjdkfd
fdkfd

"""

# %%
#  여러줄 주석

'''
   여러줄 주석
   여러줄 주석은 나중에 문자열 표현에서도 사용됨. 
'''



###02-02 출력 / 문자열 

## print(출력값, 출력값값) 
print('hello')
print("hello")
#여러개의 데이터 화면에 출력
print(10,20,30,40,50)
#문자열을 여러번 출력
print("abc"*3)
print('abc'*3)


#문자열+숫자 안됨.
print("학번:"+100) #오류발생
print("학번:",100)
print("학번:"+"100") #문자열간의 +연산은 가능함
print("학번:"+str(100)) #str() : 문자열로 변환
print("'안녕하세요' 라고 말했습니다.")
print('"안녕하세요" 라고 말했습니다.')


 
## print(출력값1, end='')
print(10,end="\t") 
print(20,end="\t")
print(30,end="\n") #end 속성값의 기본값 : \n.
#10,20,30 출력
print(10,end=",")
print(20,end=",")
print(30) 
 
 

# 들여쓰기 오류 
# print("문장의 처음에 공백이 있으면 오류") #오류 

### print( 문자열  % )
#형식문자를 이용하여 출력하기 : 
#  %d(10진수정수)
#  %f(실수)
#  %s(문자열)
print("%d * %d = %d" % (2.0,3,6))  # 2 * 3 = 6
print("%f * %f = %f" % (2,3,6))    # 2.000000 * 3.000000 = 6.000000
print("%.2f * %.2f = %.2f" % (2,3,6)) # 2.00 * 3.00 = 6.00
# %x,%X : 16진수 표현  0~ 9, a,b,c,d,e,f  
print("%X" % (255))  #FF  F * 15 + 15 
print("%x" % (255))  #ff  
print("안녕 %s!, 나도  %s" % ("홍길동","김삿갓"))
print("%X" % (255))  #FF
print("%x" % (255))  #ff
print("안녕 %s!, 나도  %s" % ("홍길동","김삿갓"))


#format 함수를 이용한 출력
#{0:5d} : 첫번째값을 정수형 5자리로 출력
#{1:5d} : 두번째값을 정수형 5자리로 출력
#{2:5d} : 세번째값을 정수형 5자리로 출력
#{2:10.2f} : 세번째값을 정수형 5자리로 출력
print("{0:5d}{1:5d}{2:5d}".format(100,200,300))
print("{1:5d}{2:5d}{0:5d}".format(100,200,300))
print("{0:5d}{1:5d}{2:5f}".format(100,200,300))
print("{0:5d}{1:5d}{2:10.2f}".format(100,200,300))
print("{1:5d}{2:5d}{0:5d}".format(100,200,300))


# print(f'{변수1} {변수2} 직접 변수이름으로 출력
a=100
b=200
print(a,b)
print(f"{a}와 {b}를 프린트 하겠습니다 ")


### 특수 문자 (Escape character)
#\" 
print("\"안녕하세요\" 라고 말했습니다.")
print('\'안녕하세요\' 라고 말했습니다.')

# \n : new line
print("동해물과 백두산이 마르고 닳도록\n하느님이 보우하사")
'''
여러줄 주석
문자열의 """ 표시는 여러줄 문자열 표시에 사용 가능함

'''

print("""동해물과 백두산이 마르고 닳도록
 하느님이 보우하사 우리나라 만세
 무궁화 삼천리 화려강산""")

print('''동해물과 백두산이 마르고 닳도록
 하느님이 보우하사 우리나라 만세
 무궁화 삼천리 화려강산''')


a='hello'
cnt=0

for i in range(len(a)):
    if a[i] == 'l':
        cnt+=1
print(a,"에서 l 문자의 갯수:",cnt)  #2

print(a,"에서 l 문자의 갯수:",a.count('l'))  #2
print(a,"에서 a 문자의 갯수:",a.count('a'))  #0

#a 문자열에서 l문자의 위치(인덱스) 출력하기
print(a,"에서 l 문자의 위치:",a.find('l'))  #2
print(a,"에서 l 문자의 위치:",a.index('l')) #2



#a 문자열에서 3번인덱스 부터 l문자의 위치(인덱스) 출력하기
print(a,"에서 l 문자의 위치:",a.find('l',3))  #3
print(a,"에서 l 문자의 위치:",a.index('l',3)) #3
#a 문자열에서 3번인덱스 부터 o문자의 위치(인덱스) 출력하기
print(a,"에서 o 문자의 위치:",a.find('o',3))  #4
print(a,"에서 o 문자의 위치:",a.index('o',3)) #4

#a 문자열에서 4번인덱스 부터 l문자의 위치(인덱스) 출력하기
print(a,"에서 l 문자의 위치:",a.find('l',4))  #-1
print(a,"에서 l 문자의 위치:",a.index('l',4)) #오류


#a 문자열에서 a문자의 위치(인덱스) 출력하기
print(a,"에서 a 문자의 위치:",a.find('a'))  #-1
print(a,"에서 a 문자의 위치:",a.index('a')) #오류. 예외처리필요


        

ss = '123'    # 숫자, 문자 또는 숫자
ss = 'Aa123'  #  문자 또는 숫자
ss = 'Aa'     #  문자 , 문자 또는 숫자
ss = 'AA'     # 문자, 문자 또는 숫자, 대문자 
ss = 'aa'     # 문자, 문자 또는 숫자, 소문자 
ss = '     '  # 공백
ss = '  aa   ' # 소문자
ss = '  AA   ' # 대문자
ss = '  Aa   '  # 없음
# %%
if ss.isdigit() :
    print(ss,": 숫자")
if ss.isalpha() :
    print(ss,": 문자")
if ss.isalnum() :
    print(ss,": 문자 또는 숫자")
if ss.isupper() :
    print(ss,": 대문자")
if ss.islower() :
    print(ss,": 소문자")
if ss.isspace() :
    print(ss,": 공백")

# %%
x=5
x=10.5
x=True
x='aaaaa'

x=10
str(100)
str=1000

sum1=100


del x

print(x)


'''
1010
0101
--------
0000 : and, &
1111 : or , |

1111 : xor

00001010 : ~10

11111111
11110101 : 1 sign bit 마이너스
========
00001010  : 10 + 1 ---> 마이너스  -11
'''  

# 기본함수
# 1. input
num = int(input("자연수를 입력하세요"))

int("11", 2) # 1 * 2 + 1
int("11", 8) # 1 * 8 + 1
int("11", 16) # 1 * 16 + 1

bin(10)
oct(10)
hex(10)


score = 65
if score >= 90 :
    print("A학점")
    print("합격입니다.")
else : 
    if score >= 80 :
        print("B학점")
        print("합격입니다.")
    else :
       if score >= 70 :
         print("C학점")
         print("합격입니다.")
       else :
        if score >= 60 :
               print("D학점")
        else :
            print("F학점")

# if elif 구문
if score >= 90 :
   print("A학점")                
   print("합격입니다.")
elif score >= 80 :
   print("B학점")                
   print("합격입니다.")
elif score >= 80 :
    print("B학점")                
    print("합격입니다.")
elif score >= 60 :
   print("D학점")                
   print("합격입니다.")
else :
   print("F학점")                
   print("불합격입니다.")     
   
   
# 반복문
#1부터 100까지의 합 구하기
num = 100
hap = 0
# range(1,num+1,증감값) : 1 ~ num까지의 숫자들
for i in range(1,num+1) :
    hap += i
print ("1부터 %d까지의 합:%d" % (num,hap))   


# 1) 1부터 100까지의 짝수의 합 구하기
hap = 0   
for i in range(2,num+1, 2) :
    hap += i
print ("1부터 %d까지의 합:%d" % (num,hap))   


hap = 0 
for i in range(1,num+1) :
    if i % 2 == 0:
        hap += i
print ("1부터 %d까지의 합:%d" % (num,hap))   




num = 1
while num <= 5 :
  print(num, end="")
  num += 1   



#break : 반복문 종료
#continue : 반복문의 처음으로 제어 이동
hap = 0
for i in range(1,11) : #1 ~ 10
   if i == 5 :
       break;
   hap += i
print('hap=',hap) # 10   


hap = 0
for i in range(1,11) :
   if i == 5 :
       continue
   hap += i
print('hap=',hap)  # 50  




# while 문으로 1부터 100까지의 합 구하기



num = 1
hap=0
while num <= 100 : 
  hap +=num
  num += 1   
print("1부터 100까지의 합 구하기",hap)


# while 문으로 1부터 100까지의 짝수의 합 구하기

num = 2
hap=0
while num <= 100 : 
  hap +=num
  num += 2   
print("1부터 100까지의 짝수 합 구하기",hap)

num = 1
hap=0
while num <= 100 : 
  if num % 2 == 0:
        hap +=num
  num += 1   
print("1부터 100까지의 짝수 합 구하기",hap)



###-----
#중첩반복문 (구구단)
i,j=0,0  #초기화 방식
for i in range(2,10) :  # 2 ~ 9
    print("%5d단" % i)
    for j in range(2,10) : # 2 ~ 9
        print("%2d X %2d = %3d" % (i,j,(i*j)))
    print()   


# 구구단 while 문으로 작성

i=2
while  i <= 9    :
    print("%5d단" % i)
    j=2
    while   j <= 9   :
        print("%2d X %2d = %3d" % (i,j,(i*j)))
        j+=1
    print()
    i+=1

'''    
1. 직각 삼각형 출력하기

*
**
***
****
*****

'''

h=5
### 1 ####
for i in range(1,h+1) :
   for j in range(1,i+1) :
       print("*",end="")
   print()    

### 2 ####
for i in range(1,h+1) :
   print("*"*i)   

'''
2. 역 직각 삼각형 출력하기

*****
****
***
**
*
'''
for i in range(h,0,-1) :
   for j in range(1,i+1) :
       print("*",end="")
   print()    

for i in range(h,0,-1) :
   print("*"*i)
   
   
###  연습문제

# 연습문제 1 : 
# 3741초가 몇시간 몇분 몇초인지 출력하기 ( 1 시간 2 분 21 초)
# input함수:키보드에서 입력받기.
second=int(input("초를 입력하세요:"))
print(second//3600,'시간', (second%3600)//60, '분', 
                     second%60, '초')
# 3741//3600 시간
# 3741%3600  초//60 분 
# 3741%60  초
# // 정수,   % 나머지 


# 연습문제 2 : 
# 금액(3650)을 입력 받아 잔돈(500,100,50,1)으로 바꿔주는 프로그램을 작성 하기 
# 답
'''
금액을 입력하세요:3650
500원 동전의 갯수 7
100원 동전의 갯수 1
50원 동전의 갯수 1
10원 동전의 갯수 0
1원 동전의 갯수 0
'''


# input함수:키보드에서 입력받기.
money = int(input("금액을 입력하세요:"))
print("500원 동전의 갯수", money//500) 
print("100원 동전의 갯수", money%500//100)
print("50원 동전의 갯수", money%100//50)
print("10원 동전의 갯수", money%50//10)
print("1원 동전의 갯수", money%10)


'''
연습문제 3
12345          
*****    1
 ****    2    
  ***    3     
   **    4    
    *    5     
'''  


for i in range(1, h+1):
    for j in range(1, h+1):
        if j < i :
          print(' ', end="")
        else:
          print("*", end="")
        
    print()


'''
이모양을 프린트 하세요
    *     
   * *    
  *   *   
 *     *  
*       * 
 *     *  
  *   *   
   * *    
    *     
'''


num=9
point1=num//2
point2=num//2
for i in range(num):
    for j in range(num):
        if j == point1 or j == point2:
            print("*", end="")
        else:
            print(" ", end="")
    print()
    # print(point1, point2)
    point1 -=1
    point2 +=1    
    if point1 == 0 :
        point1 = num-1
        point2 = 0
    
    














