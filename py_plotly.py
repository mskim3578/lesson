import plotly.express as px
import pandas as pd
import numpy as np

# 임의의 시계열 FDC 샘플 데이터 생성
np.random.seed(42)
times = pd.date_range(start='2026-06-30 00:00', periods=100, freq='min')
pressure_values = 10 + np.random.normal(0, 0.5, 100) # 평균 10, 약간의 변동성

df_sample = pd.DataFrame({'Time': times, 'Pressure': pressure_values})

# Plotly 선 그래프 그리기
fig = px.line(df_sample,  x='Time',  y='Pressure', title='실시간 챔버 압력(Pressure)')
# 그래프 레이아웃 세부 설정 (마우스 가이드선 추가)
fig.update_traces(mode='lines+markers') # 선과 점을 동시에 표현
fig.update_layout(hovermode='x unified') # 마우스를 올리면 해당 시간대의 값이 정렬되어 표시됨
# 그래프 출력
fig.write_html('sampleplotly.html')
