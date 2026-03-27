import pandas as pd

import matplotlib.pyplot as plt 
import numpy as np
plt.rcParams['font.family'] = 'Malgun Gothic'   # Windows


# 마이너스 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

def sample_scatter():  
    fig=plt.figure(figsize=(10, 8))
    x1 = np.linspace(0, 10, 100)  # 10포함 한다
    y1 = np.random.rand(100)   
    
    # plt.plot(x1, y1, color='b', linestyle='-', marker='o', label="plt.plot()")
    plt.plot(x1 , y1, color='b', linestyle='-', label="plt.plot()")
    plt.title("Line Plot 그래프")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.yticks([x/10 for x in range(11)])
    plt.legend()
    plt.show()

    # plt.show()    
    return fig
