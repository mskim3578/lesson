# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 15:34:21 2026

@author: letuin
"""

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

###  pip install spyder-kernels==3.1.*
### 지역변수(local), 전역변수(global)

def func3():
    c=300 # 지역변수
    print(a,b,c)


def func4():
    a=110
    b=220
    print(a,b)

def func5():
    global a,b
    a=1  #global
    b=2  # global
    print(a,b)

func3()
func4()
func5()

# 매개변수

def add(v1, v2):
    return  v1+v2

def sub1(v1, v2):
    return v1-v2 

hap=add(10,20)
sub=sub1(20, 10)


hap=add(10.5,20.1)
sub=sub1(10.5, 20.1)


# 가변 매개 변수

def multiparam(* a):
    result = 0 
    for i in a:
        result +=i 
    return result
        
multiparam()
multiparam(10)
multiparam(10, 20)
multiparam(10, 20, 30)
multiparam(1.5, 2.5, 3 )

multiparam("1.5", 2.5, 3)

# 딕션너리 가변인자 **kwargs
def dictDefine(**kwargs):
    print("="*30)
    print(kwargs)
    for k in kwargs:
        print(k, ":", kwargs[k])
    print("\n 총길이", len(kwargs))
        
# 함수 호출
dictDefine()
dictDefine(a='apple', b='banana', c='carrot')
dictDefine(n='nano', u='umbrella', m='moutain', s='sweet', d='dress')


# 매개 변수 기본값 설정

def hap1(num1=0, num2=1):
    return num1+num2
hap1()
hap1(10)
hap1(0, 10)

######################################
# 연습문제 1
# 리스트의 평균을 구하는 함수 getMean() 를 작성 하세요 
list1 = [1,2,3,4,5,6]

def getMean(li):
    if len(li) > 0 :
        return sum(li)/len(li)
    else:
        return 0
    

getMean(list1)

####   lambda function
(lambda li: sum(li)/len(li) if len(li) > 0 else 0)(list1)

getMean([])


# 연습문제 2
# 계산기호 인자 값에 따라 뒤에 가변인자를 계산하는 
# 함수 calChoice(choice, *args) 를 구현 하세요
# 함수 calChoice2(choice, first, *args) 를 구현 하세요

def calChoice(choice, *args):
    if choice == '+':
       results=0
       for i in range(len(args)):
           results +=args[i]
       # print(results)  
    elif choice == '/':
       results=1
       for i in range(len(args)):
           results /=args[i]
       # print(results)
    elif choice == '*':
       results=1
       for i in range(len(args)):
           results *=args[i]
       # print(results) 
    elif choice == '-':
       results=0
       for i in range(len(args)):
           results -=args[i]
       # print(results)
    return results

calChoice('+', 1,2,3,4)
calChoice('-', 1,2,3,4)
calChoice('*', 1,2,3,4)
calChoice('/', 1,2,3,4)


def calChoice2(choice, first, *args):
    
    for i in range(len(args)):
        
        if choice == '+': 
               first +=args[i]
        elif choice == '/':
               first /=args[i]
        elif choice == '*':
               first *=args[i]
        elif choice == '-':
               first -=args[i]
        
    return first


calChoice2('+', 0,2,3,4)
calChoice2('-', 1000,2,3,4)
calChoice2('*', 1,2,3,4)
calChoice2('/', 1000,2,3,4)



# 연습문제 3
# disctionary 자료를 입력 받아 maxValue(dics) 함수를 만들고 
# 과일: 남은갯수 일때 가장 많이 남은 과일이랑 갯수를 프린트 하세요

dics = {'apple': 10, 'banana': 25, 'cherry': 25, 'orange':15}
def maxValue(dics):
    
    tmax = max(dics.values())
    maxli = [ (k, v) for k, v in dics.items()  if dics[k] == tmax]
    return maxli



maxValue(dics)  #답 [('banana', 25), ('cherry', 25)]




# 연습문제 4
'''
딕셔너리 값(Value)들의 총합과 평균 구하기
문제: 과이름과 가격(또는 점수)이 Key: Value 형태의 딕셔너리로 주어질 때, 
모든 Value의 총합과 평균을 계산하여 
딕셔너리로 반환하는 함수 analyze_dict(scores)를 작성하세요. 
(반환 형태: {'total': 합계, 'average': 평균})
'''
# 테스트
data = {'math': 80, 'eng': 90, 'kor': 85}
def analyze_dict(data):
    if len(data) == 0:
        return {'total': 0, 'average': 0}
    
    total = sum(data.values())
    avg= total/ len(data)
    return {'total': total, 'average': avg}

###   lambda 구현
print(analyze_dict(data))  # 출력: {'total': 255, 'average': 85.0}
print(analyze_dict({}))  # 출력: {'total': 255, 'average': 85.0}

tlamb = lambda data : {'total': sum(data.values()), 'average': sum(data.values())/len(data)} \
                       if len(data) > 0 else {'total': 0, 'average': 0}
print(tlamb(data))    
print(tlamb([]))     
    
    
'''
좌표 튜플들의 거리 계산하기
문제: 2차원 좌표를 나타내는 튜플((x, y))들이 담긴 리스트를 파라미터로 받아, 
원점(0, 0)에서 가장 가까운 좌표를 반환하는 
함수 get_closest_point(points)를 작성하세요. 
(피타고라스 정리에 따른 거리 비교, 제곱 값으로 비교 가능)
'''

points=[(3, 4), (1, 1), (5, 5)]

def dictance_pro(p):
    return p[0]**2 + p[1]**2
    

def get_closest_point(li1):
    # return min(points,  key=dictance_pro)
    return min(points,  key=lambda p:p[0]**2 + p[1]**2)

print(get_closest_point(points))  # 출력: (1, 1)



# 람다식으로 함수 구현

hap2 = lambda num1, num2 : num1+num2
hap3 = lambda num1=0, num2=2 : num1+num2
# def hap2(num1, num2):
#     return num1+num2

print(hap2(10,20))
print(hap3())



print((lambda num1, num2 : num1+num2)(10,20))
print((lambda num1=0, num2=2 : num1+num2)())

# 람다 함수를 이용한 map

mylist = [1,2,3,4,5]
add = lambda num : num * 10
mylist = list(map(add, mylist))

mylist = [1,2,3,4,5]
mylist = list(map((lambda num : num -10), mylist))
print(mylist)


################################################

# 복수 리스트에 적용하는 람다 함수
list1=[1,2,3,4]
list2=[10,20,30,40]
list3=[10, 20, 30, 40, 50]

hap=lambda n1, n2 : n1 + n2
haplist = list(map(hap, list1, list2))

haplist = list(map(lambda n1, n2 : n1 + n2, list1, list2))

list1.append(0)
list2.append(0)
haplist = list(map(lambda n1, n2, n3 : n1+n2+n3, list1, list2, list3))

'''
4개의 리스트 각 아이템 요소가 '-'로 연결되어 출력되도록 하여라
최종 결과는 리스트로 저장되어야 하며 map() 함수를 활용한다
'''

num_list = [100, 200, 300, 400]
name_list = ['길동', '동미', '미영', '영철']
gender_list = ['남','여','여','남']
address_list = ['서울','대전','부산','대구']

#['100-길동-남-서울', '200-동미-여-대전', '300-미영-여-부산', '400-영철-남-대구']

list(map(lambda a,b,c,d : f'{a}-{b}-{c}-{d}', 
             num_list, name_list, gender_list, address_list))

# ['100-길동-남-서울', '200-동미-여-대전', '300-미영-여-부산', '400-영철-남-대구']




