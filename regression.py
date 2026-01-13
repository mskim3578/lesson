import openpyxl 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import warnings

# 모든 FutureWarnings 무시
# warnings.simplefilter(action='ignore', category=FutureWarning)  
warnings.simplefilter(action='ignore', category=FutureWarning)



# linear 알고리즘  임포트
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
# pip install xgboost lightgbm 


# %%  렛유인 project1 function

plt.rcParams['axes.unicode_minus'] = False  #chart에 마이너스 프린트 
plt.rc("font",family="Malgun Gothic")   # 한글 프린트   

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "DecisionTree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

