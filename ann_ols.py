import numpy as np
import pandas as pd
from scipy import stats
#  pip install scipy
import matplotlib.pyplot as plt
#  pip install matplotlib
import seaborn as sns
#  pip install seaborn
plt.rcParams['axes.unicode_minus'] = False
plt.rc("font",family="Malgun Gothic")

# %%
# 1. 가상 데이터 준비 (설비 A, B, C의 수율 데이터)
group_A = np.array([85, 88, 89, 86, 90])
group_B = np.array([92, 89, 94, 93, 91])
group_C = np.array([80, 83, 82, 85, 81])
# 2. 일원분산분석(One-way ANOVA) 실행
f_stat, p_value =stats.f_oneway(group_A, group_B, group_C)

# 3. 결과 출력
print("=== Scipy ANOVA 결과 ===")
print(f"F-통계량 (F-statistic) : {f_stat:.4f}")
print(f"p-값 (p-value)          : {p_value:.6f}")
# 4. 유의수준 0.05 기준으로 판정
alpha = 0.05
if p_value < alpha:
    print("=> 판정: p-value가 0.05보다 작으므로 귀무가설 기각. 세 그룹의 평균 수율은 통계적으로 유의미한 차이가 있다.")
else:
    print("=> 판정: p-value가 0.05보다 크므로 귀무가설 채택. 세 그룹의 평균 수율 차이가 있다고 볼 수 없다.")
'''
=== Scipy ANOVA 결과 ===
F-통계량 (F-statistic) : 29.6923
p-값 (p-value)          : 0.000023
=> 판정: p-value가 0.05보다 작으므로 귀무가설 기각. 세 그룹의 평균 수율은 통계적으로 유의미한 차이가 있다.
'''
# %%
########  ols
import statsmodels.api as sm
import statsmodels.formula.api as smf

def eda_pro(df, col): 
    fig, ax = plt.subplots(2, 2, figsize=(5, 5))
    ax[0, 0].plot(range(len(df)), df[col])
    #----
    ax[0, 0].axhline(y=df[col].mean(), color='blue')   
    ax[0, 0].axhline(y=df[col].mean()+(3*df[col].std()), 
                     color='r')  
    ax[0, 0].axhline(y=df[col].mean()-(3*df[col].std()), 
                     color='r')
    #----
    ax[0, 0].set_title("Line Plot")

    ax[0, 1].bar(range(len(df)), df[col])
    ax[0, 1].set_title("Bar Chart")
    
    ax[1, 0].hist(df[col], bins=10, color='purple', 
                  edgecolor='black', alpha=0.7)
    #----
    ax[1, 0].axvline(x=df[col].mean(), color='blue')   
    ax[1, 0].axvline(x=df[col].mean()+(3*df[col].std()), 
                     color='r')  
    ax[1, 0].axvline(x=df[col].mean()-(3*df[col].std()), 
                     color='r')
    #----
    ax[1, 0].set_title("Histogram")

    ax[1, 1].scatter(range(len(df)), df[col])
    ax[1, 1].axhline(y=df[col].mean(), color='blue')
    ax[1, 1].set_title("Scatter Plot")

    plt.tight_layout()
    plt.show()
  
    
# pip install statsmodels

df_sample=pd.read_csv('data_folder/lot_sample_1000.csv')
df_sample.columns

eda_pro(df_sample, 'PECVD_Thick')

x=df_sample[df_sample.columns[4:-1]]

tcorr=df_sample[df_sample.columns[3:-1]].corr()
sns.heatmap(tcorr,annot=True, cmap='coolwarm')
plt.show()

# y target
y1=df_sample['PECVD_Thick']
y2=df_sample['Pass_Fail'].map({'Pass':1, 'Fail':0})

'''
Index(['Lot_ID', 'Batch_ID', 'Wafer_ID', 'PECVD_Thick',
       'Etch_Rate', 'Idsat',
       'BVdss', 'Vth', 'Defect_Count', 'Yield_Percentage', 'Pass_Fail'],
      dtype='str')

'''

# 3. OLS 모델 생성 및 학습 (주의: y가 첫 번째, X가 두 번째 인자)

x=sm.add_constant(x)
model=sm.OLS(y1,x).fit()

# 4. 분석 결과 요약 리포트 출력
print("==== [2] OLS 모델 분석 결과 리포트 ====")
print(model.summary())

'''

print(model.summary())
                            OLS Regression Results                            
==============================================================================
Dep. Variable:            PECVD_Thick   R-squared:                       0.009
Model:                            OLS   Adj. R-squared:                  0.003
Method:                 Least Squares   F-statistic:                     1.448
Date:                Wed, 12 Aug 2026   Prob (F-statistic):              0.193
Time:                        16:17:18   Log-Likelihood:                -4948.4
No. Observations:                1000   AIC:                             9911.
Df Residuals:                     993   BIC:                             9945.
Df Model:                           6                                         
Covariance Type:            nonrobust                                         
====================================================================================
                       coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------
const             1527.9990     93.573     16.329      0.000    1344.375    1711.623
Etch_Rate           -0.1164      0.091     -1.283      0.200      -0.294       0.062
Idsat                0.0314      0.044      0.711      0.477      -0.055       0.118
BVdss               -0.9616      2.645     -0.364      0.716      -6.151       4.228
Vth                -23.5967     21.844     -1.080      0.280     -66.462      19.268
Defect_Count         1.0713      0.944      1.135      0.257      -0.781       2.924
Yield_Percentage     0.2669      0.744      0.359      0.720      -1.194       1.727
==============================================================================
Omnibus:                        0.395   Durbin-Watson:                   2.007
Prob(Omnibus):                  0.821   Jarque-Bera (JB):                0.300
Skew:                          -0.031   Prob(JB):                        0.861
Kurtosis:                       3.058   Cond. No.                     6.89e+04
==============================================================================                    1.867   Cond. No.                     6.89e+04
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 6.89e+04. This might indicate that there are
strong multicollinearity or other numerical problems.

'''
########################  matplot ex
import pandas as pd
df_age=pd.read_csv('data_folder/age.csv', encoding='cp949',
                   thousands=",", index_col=0)

# 1. columns 확인
df_age.columns
cols_10= df_age.columns[3:13]
df_age.info()

df_10=df_age[cols_10]
df_10.info()




#2.  모든컬럼의 내용을  1세, 2세..... 로 수정 하세요
new_col = [ x.split('_')[2] for x in cols_10]
df_10.columns=new_col

plt.bar(range(10), df_10.sum())
plt.show()

#3. x : 1 ~10, x col

'''
<class 'pandas.DataFrame'>
Index: 3910 entries, 서울특별시  (1100000000) to 제주특별자치도 서귀포시 예래동(5013062000)
Columns: 103 entries, 2025년06월_계_총인구수 to 2025년06월_계_100세 이상
dtypes: int64(103)
memory usage: 3.3 MB
'''

