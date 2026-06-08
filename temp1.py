import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
plt.rcParams['axes.unicode_minus'] = False
plt.rc("font",family="Malgun Gothic")

# %%
###########
# 전세계 음주 데이터 분석 하기
drinks = pd.read_csv("data/drinks.csv")
drinks.info()
'''
  country : 국가명
  beer_servings : 맥주소비량
  spirit_servings : 음료소비량
  wine_servings : 와인소비량   
  total_litres_of_pure_alcohol : 순수 알콜량
  continent : 대륙명
'''
drinks.head(10)

# 변수=컬럼=피처
#1. beer_servings, wine_servings 
# 피어슨 상관계수  : 기본

corr1 = drinks[['beer_servings', 'wine_servings']] \
         .corr()
# kendall sample 사이즈가 작을때 
corr2 = drinks[['beer_servings', 'wine_servings']] \
         .corr(method='kendall')

# spearman 정규화 되지 않은 자료 
corr3 = drinks[['beer_servings', 'wine_servings']] \
         .corr(method='spearman')

######

drinks.corr()  #object column 때문
drinks.info()

cols=drinks.columns[1:-1]
corr1=drinks[cols].corr()

sns.heatmap(corr1, cmap="Blues", 
            fmt='f', 
            annot=True,
            # cbar=False,
            linewidth=3)
plt.show()

#seaborn 모듈의 산점도을 이용하여 시각화 하기
sns.pairplot(drinks[cols])
plt.show()


# 각변수의 결측값 갯수 조회하기
drinks.info()

drinks.isnull().sum()
drinks['continent']=drinks['continent'].fillna('OT')

# 대륙별 국가의 갯수를 파이그래프로 출력하기

drinks['country'].value_counts()
conts=drinks['continent'].value_counts()

explode=(0,0,0.1,0,0,0)
plt.pie(drinks['continent'].value_counts(),
        labels=conts.index,
        autopct="%5d%%",
        explode=explode,  # 해당 값의 파이를 빼서 표현
        shadow=True
        )

plt.title("null data to 'OT'")
plt.show()

# 1 대륙별:continent  total_litres_of_pure_alcohol 섭취량 평균
drinks.info()

def continent_pro(col):    
    cont_mean=drinks.groupby('continent')[col].mean()
    total_mean= drinks[col].mean()    
    # 2 대륙명    
    continents=cont_mean.index.tolist()
    continents.append('Mean')    
    x_pos = np.arange(len(continents))
    # y축 데이터
    y_pos = cont_mean.tolist()
    y_pos.append(total_mean)    
    bar_list = plt.bar(x_pos, y_pos , align='center',
                       alpha=0.5)
    plt.axhline(y=total_mean , color='r', linestyle='--')
    # plt.axvline(x=3 , color='b', linestyle='--')
    bar_list[len(continents)-1].set_color('r')
    plt.xticks(x_pos, continents)
    plt.ylabel(col)
    plt.title('대륙별 평균 알콜 섭취량')
    plt.show()

continent_pro('beer_servings')
continent_pro('spirit_servings')
continent_pro('wine_servings')

#대한민국은 얼마나 술을 독하게 마시는 나라인가?
# total_servings : 전체 주류 소비량 컬럼 추가

drinks["total_servings"] =\
    drinks["beer_servings"] + \
    drinks["spirit_servings"] +\
    drinks["wine_servings"]

#alcohol_rate : 알콜비율 (알콜섭취량/전체주류소비량) 추가
drinks["alcohol_rate"] = \
    drinks["total_litres_of_pure_alcohol"]/drinks["total_servings"]

drinks["alcohol_rate"]=drinks["alcohol_rate"].fillna(0)
drinks.info()

#alcohol_rate의 값으로 내림차순 정렬하기. alcohol_rate_rank 저장
alcohol_rate_rank=drinks.sort_values(by='alcohol_rate', ascending=False)\
    [['country', 'alcohol_rate']]


alcohol_rate_rank.head(20)

country_list=alcohol_rate_rank.country.tolist()
# x축 값
x_pos = np.arange(len(country_list))

# y 축값
y_pos=alcohol_rate_rank.alcohol_rate.tolist()

plt.bar(x_pos, y_pos)

