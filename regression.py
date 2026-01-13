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
