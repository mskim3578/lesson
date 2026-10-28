# -*- coding: utf-8 -*-
a=[0,0,0,0] #[10,20,30,40]
b=[]
print(a,len(a))      #len(a) : 리스트 요소의 갯수
print(b,len(b))

hap=0
for i in range(len(a)): #0,1,2,3
    a[i]=int(input(str(i+1) + '번째 숫자 입력:'))
    hap += a[i]
print(hap)
print(sum(a))



# %s : 문자열을 지정하는 형식지정문자
#pop : LIFO(stack) 관련 함수.  
#sort : 정렬 함수.

mylist = [30,10,20]
print("리스트:%s" % mylist)  # 리스트:[30, 10, 20]
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




a=10
b=a

l1 = [1,2,3]
l2=l1.copy()
l2.append(5)



###  list comprehension
numbers=[]
for n in range(1,11) : #1~10
    numbers.append(n)
print(numbers)

print([n for n in range(1,11)])


print([    n    for n in range(1,11)])


###--------------Set Comprehension
#1 ~10사이의 짝수 제곱값을 가진 set 객체 생성하기
sets=set()
for n in range(2,11,2) : #1~10
    sets.add(n*n)
print(sets)

sets = {x*x for x in range(1,11) if x%2==0}
print(sets)

set1 = {x*x for x in range(2,11,2)}
print(sets)
print(sorted(sets)) #리스트



#2의 배수이고, 3의 배수인 값만 리스트에 추가하기.
list21=[x for x in range(1,11) if x%2==0 and x%3==0]
print(list21)
list21=[x for x in range(1,11) if x%2==0 if x%3==0]
print(list21)




# 중첩 comprehension
matrix=[[1,2,3],[3,4,5],[6,7,8]]

list1 = [x for row in matrix for x in row]
print(list1)


# dictionary comprehension
nums = [1,2,3,4,5]
result = {num: 'even' if num%2 == 0 else 'odd' for num in nums}
print(result)


dic1={}
for num in nums:
  if num%2 == 0:
      dic1[num]='even'
  else:
      dic1[num]='odd'
print(dic1)


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



###----------------int 함수 이용
before = ["2024","11","08"]
print(type(before[0])) # 문자열 <class 'str'>
print(before)  # ['2024', '11', '08']
#before의 요소 각각을 int 형변환한 후 다시 list 객체로 생성 
after = list(map(int,before))
print(type(after[0])) #<class 'int'>
print(after)   #[2024, 11, 8]

print([ int(x)   for x in before])

# 일반 함수

dataList = [x for x in range(1,11)]

def evenNumber(dataList):
    resList=[]
    for i in dataList:
        if i % 2 ==0:
            resList.append(i)
    return resList

print(evenNumber(dataList))




def even_pro(n):
    return n % 2 == 0

print(list(filter(even_pro, dataList)))

print([ n    for n in dataList  if  n % 2 == 0])

############################################################################
########   연습문제 (for, comprehension, map, filter)
# 문제 1
clist=['black', 'white']
slist=['S','M','L']
# 출력 : [['black', 'S'], ['black', 'M'], ['black', 'L'], ['white', 'S'], ['white', 'M'], ['white', 'L']]

#for
sumlist=[]
for x in clist:
    for y in slist:
        sumlist.append([x,y])
        
print(sumlist)

#comprehension
[[x,y] for x in clist for y in slist]


# 문제 2
# ''의 모든 요소를 대문자로 변환한 'uppercased_fruits'를 생성하세요.,

# 출력 :  ['MANGO', 'KIWI', 'GUAVA']
fruits = ["mango", "kiwi", "guava"]
uppercased_fruits=[]
for s in fruits:
    uppercased_fruits.append(s.upper())
    
print(uppercased_fruits)

[s.upper() for s in fruits]

def map1(s):
    return s.upper()

list(map(map1, fruits))









# 문제 3
# names에서 이름 길이가 5자 이상인 것만 대문자로 바꾸어 저장하세요.
names = ["Alex", "Beth", "Caroline"]


# for   
newnames=[]    
for s  in names:
    if len(s)>5:
      newnames.append(s.upper())
print(newnames) 

# comprehension  
[ s.upper() for s in names if len(s)>5]


def map2(s):
    return len(s)> 5

list(filter(map2, names))


list(map(map1, list(filter(map2, names))))

import pandas as ps
import  streamlit as st 

###############  과제
'''
문제 : 1. 궁합음식의 키를 입력받아 해당되는 음식을 출력하기
         등록안된 경우 오류 발생. => 등록여부 판단필요
       2.종료 입력시 등록된 내용 출력하기
         등록된음식 :
              떡볶이 : 오뎅
              짜장면 : 단무지
       3. 등록이 안된경우 
          등록여부를 입력받아, 등록하는 경우 궁합음식을 입력받기  
          등록하시겠습니까(y)? 
             y입력 : foods객체에 추가.
                     궁합음식 입력받아서 foods에 추가함
             y가아닌경우 :
                     음식을 다시 입력하기    
              
''' 

foods = {"떡볶이":"어묵","짜장면":"단무지","라면":"김치","맥주":"치킨"}










