# -*- coding: utf-8 -*-
"""
Created on Wed May 27 09:34:03 2026

@author: user
"""

### 조건문 : if문
# 들여쓰기 해야함
# %%
score = 85
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
   
score = 65
# 한줄 if 문
if score >= 60 : print('PASS')  


# True if 조건식 else False

print (score, '점수는', 'PASS' if score >=60 else 'FAIL')


# 반복문
#1부터 100까지의 합 구하기
num = 100
hap = 0
# range(1,num+1,증감값) : 1 ~ num까지의 숫자들
for i in range(1,num+1) :
    hap += i
print ("1부터 %d까지의 합:%d" % (num,hap))   


num = 1
while num <= 5 :
  print(num, end=" ")
  num += 1      

   
#  1 ~ 100까지 짝수의 합을 구하세요

######   for
num = 100
hap = 0
# range(2,num+1,증감값) : 2,4,6  ~ num까지의 숫자들
for i in range(2,num+1, 2) :
    hap += i
print ("1부터 %d까지의 합:%d" % (num,hap)) 


hap = 0
# range(1,num+1,증감값) : 1 ~ num까지의 숫자들
for i in range(1,num+1) :
    if i % 2 == 0:
        hap += i
print ("1부터 %d까지의 합:%d" % (num,hap))   

#### while
num = 0
hap=0
while num <= 100 :
  print(num, end=" ")
  hap +=num
  num += 2  
print ("1부터 %d까지의 합:%d" % (num,hap)) 


num = 0
hap=0
while num <= 100 :
  print(num, end=" ")
  if num % 2 == 0:
      hap +=num
  num += 1  
print ("1부터 %d까지의 합:%d" % (num,hap)) 



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


###-----
#중첩반복문 (구구단)
i,j=0,0  #초기화 방식
for i in range(2,10) :  # 2 ~ 9
    print("%5d단" % i)
    for j in range(2,10) : # 2 ~ 9
        print("%2d X %2d = %3d" % (i,j,(i*j)))
    print()   


###---
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



###-----
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


#######  자료 구조 (collection)
##################################################리스트

###---  정수형 리스트
a=[0,0,0,0] #[10,20,30,40]
b=[]
print(a,len(a))      #len(a) : 리스트 요소의 갯수
print(b,len(b))

#a 리스트의 길이 만큼 숫자를 입력받아, a에 저장하고, 입력받은 수의
# 전체 합계를 출력하기
# for 문만 먼져실행행
hap=0
for i in range(len(a)) :#i: 0 ~ 3
    a[i] = int(input(str(i+1)+'번째 숫자 입력: '))
    hap += a[i]
print(a,"요소의 합:",hap) 
print(a,"요소의 합:",sum(a))  #sum(리스트) : 리스트요소의합  




# 1번째 숫자 입력: 1
# 2번째 숫자 입력: 2
# 3번째 숫자 입력: 3
# 4번째 숫자 입력: 4
# [1, 2, 3, 4] 요소의 합: 10
# [1, 2, 3, 4] 요소의 합: 10
# %%
#a 리스트의 길이 만큼 숫자를 입력받아, 
# b에 저장하고, 입력받은 수의 합계 출력하기
b=[]
hap=0
for i in range(len(a)) :#i: 0 ~ 3
    temp= int(input(str(i+1)+'번째 숫자 입력: '))
    b.append(temp)
    hap += b[i]
print(b,"요소의 합:",hap) 
print(b,"요소의 합:",sum(b))  #sum(리스트) : 리스트요소의합  

# %s : 문자열을 지정하는 형식지정문자
#pop : LIFO(stack) 관련 함수.  

#sort : 정렬 함수.

mylist = [30,10,20]
print("리스트:%d" % mylist)  # 리스트:[30, 10, 20]
mylist.append(40)
print("mylist.append(40) 리스트:%s" % mylist) # mylist.append(40) 리스트:[30, 10, 20, 40]
print("pop() 메서드 결과:%s" % mylist.pop())  #pop() 메서드 결과:40
print("pop() 메서드 후 리스트 :%s" % mylist)   # pop() 메서드 후 리스트 :[30, 10, 20]
mylist.sort()
print("mylist.sort() 후 리스트 : %s" % mylist)   # mylist.sort() 후 리스트 : [10, 20, 30]
mylist.reverse() #역순 재배치
print("mylist.reverse() 후 리스트 : %s" % mylist) # mylist.reverse() 후 리스트 : [30, 20, 10]



###------튜플 사용하기
tp1=(10,20,30)
print(tp1)
#tp1.append(40)     #tuple은 변경 할 수 없음
list1 = list(tp1)   #tuple을 list로 변경 
list1.append(40)    #list 객체에 요소 추가 
tp1 = tuple(list1)  # list를 tuple로 변경
print(tp1)
print("tp1의 크기=",len(tp1))
print("tp[1:3]=",tp1[1:3])
print("tp[:3]=",tp1[:3])
print("tp[2:]=",tp1[2:])
print("tp[::2]=",tp1[::2])
print(tp1[0],tp1[1],tp1[2])    #인덱스를 이용하여 접근 가능
a,b,c,d=tp1                    #tuple의 각 요소를 각각의 변수에 저장
print(a,b,c,d)

###------- 튜플 연산
my_tuple = (1, 2, 3)
print(my_tuple*3)  # (1, 2, 3, 1, 2, 3, 1, 2, 3)
print(my_tuple + my_tuple)  # (1, 2, 3, 1, 2, 3)



# Named Tuple => 반드시 1:1 대응!
student = (name, age, gender) = ('홍길동', 27, '남') 
# ('제인', 27, '여', True)의 경우 1:1이 아니라서 안 됨

print(f'학생 정보 = {student}', type(student))  # 학생 정보 = ('제인', 27, '여') <class 'tuple'>
print(f'이름 = {name}', type(name))   #  이름 = 제인 <class 'str'>
print(f'나이 = {age}', type(age))     #  나이 = 27 <class 'int'>
print(f'성별 = {gender}', type(gender))   #성별 = 여 <class 'str'>


score_dic = {'lee':100, 'hong':70, 'kim':90}
print(score_dic)
type(score_dic)


# value 
t1=score_dic.values()
t1=list(score_dic.values())

# 키와 값의 쌍으로 프린트
t1=score_dic.items()
t1=list(score_dic.items())

# 반복문 표현
for key in score_dic:  #keys()
   print(key, score_dic[key])
   
for key in score_dic.keys():  
   print(key, score_dic[key])
   
for key, val in score_dic.items(): 
   print(key, val)

for val in score_dic.values():  
   print(val)



   
# set
set1 = {1,2,3,4,5,6}
set2 = {1,2,3,4,5,1,2,3,4,5}
set3 = {5,6,7,8}

# 교집합
print('set1과set2의 교집합 ', set1 & set2) #set1과set2의 교집합  {1, 2, 3, 4, 5}
print('set1과set3의 교집합 ', set1 & set3) # set1과set3의 교집합  {5, 6}
print('set1.intersection(set3)의 교집합 ', set1.intersection(set3)) 
#set1.intersection(set3)의 교집합  {5, 6}

# 합집합
print('set1과set2의 합집합 ', set1 | set2) #{1, 2, 3, 4, 5, 6}
print('set1과set3의 합집합 ', set1 | set3) # {1, 2, 3, 4, 5, 6, 7, 8}
print('set1.union(set3)의 합집합 ', set1.union(set3)) # {1, 2, 3, 4, 5, 6, 7, 8}




# 자료구조 객체 초기화

li1 = []
dic1 = {}
tu1 = ()
set1 = set()

a=1 
b=a
b=100

a=[]
b=a
b.append('1')
a.append('a')

# %%

############################ comprehension(컴프리헨션) 방식으로 Collection 객체 생성
# 규칙성이 있는 데이터를 Collection 객체의 요소로 저장하는 방식
# numbers 리스트 : 1 ~ 10까지의 데이터 저장

### list Comprehension

numbers=[]
for n in range(1,11) : #1~10
    numbers.append(n)
print(numbers)

#컴프리헨션 이용
print(x for x in range(1,11))
clist=[x for x in range(1,11)]
print(clist)



#2의 배수이고, 3의 배수인 값만 리스트에 추가하기.
list21=[x for x in range(1,11) if x%2==0 and x%3==0]
print(list21)
list21=[x for x in range(1,11) if x%2==0 if x%3==0]
print(list21)



set2 = {1,4,5,6,7,}
s1=sorted(set2)

# 중첩 comprehension
matrix=[[1,2,3],[3,4,5],[6,7,8]]

list1 = [x for row in matrix for x in row]
print(list1)






