import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
plt.rcParams['axes.unicode_minus'] = False
plt.rc("font",family="Malgun Gothic")





#자동차 연비데이터의 mpg 값을 히스토그램으로 출력하기
df = sns.load_dataset("mpg")
df.info()
#DataFrame plot 히스토그램 출력
df["mpg"].plot(kind="hist")
plt.show()

#간격을 20개로 분리한 히스토그램 출력


df["mpg"].plot(kind="hist",bins=20,color='coral',\
               figsize=(10,5),histtype='bar', width=1)
plt.title("MPG 히스토그램")
plt.xlabel("mpg(연비)")
plt.show()

df["mpg"].min()  # 9.0
df["mpg"].max()  # 46.6
