import  matplotlib.pyplot  as plt
# pip install matplotlib
import  numpy as np
import cv2

def getMean(li):
    if len(li) > 0 :
        return f'평균은 {sum(li)/len(li)}'
    else:
        return f'평균은 0'
    
    
def calChoice(choice, first, *args):
        if choice  in '-+/*' :
            for i in range(len(args)):
                if choice == '+':
                    first +=args[i]
                elif choice == '-':
                    first -=args[i]
                elif choice == '*':
                    first *=args[i]
                elif choice == '/':
                    first /=args[i]
            # print(first) 
            return f' 결과는 {first}'
        else :
            return f'{choice} 연산자가 아닙니다'
        
def bar_pro():
    
    x=np.linspace(0, 10, 10)
    y=np.random.rand(10)
    
    plt.bar(x,y, color='skyblue', edgecolor='black')
    plt.title("Bar Chart")
    
    return plt
    
    
    
def maxValue(dics):
    
    tmax = max(dics.values())
    maxli = [  (d,v)  for d, v in dics.items() \
                             if dics[d] == tmax]
    return f'{maxli}'





