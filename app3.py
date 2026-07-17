import streamlit as st
import  aifunct  as ai
from streamlit.web import cli as stcli
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


menus=[ "1. 데이터 탐색",
        "2. Line Chart",
        "3. Histo Chart",
        "4. 분류 모델",
        "5. 회기 모델",
        "6. 우량, 분량 예측",
        "7. 예측 Thickness "       
       ]
@st.cache_data
def get_csv_dataset():
    df_ai=ai.load_data()
    return df_ai
    
df_ai=get_csv_dataset()


def main():
    menu = st.sidebar.radio('Project3 : 가상 계측(VM) 및 이상 감지(FDC)', menus)
    
    if menu == menus[0]:
       # st.dataframe(df_ai)
       cols=df_ai.columns.to_list()
       chk_col = st.radio("컬럼을 선택하세요", cols, horizontal=True, index=0) 
       
       temp_li= df_ai[chk_col].unique().tolist()
       
       if len(temp_li) > 20:
           result = f'{temp_li[:20]}.... {len(temp_li)}'
       else :
           result = f'{temp_li} : {len(temp_li)}'
           
       st.write(result)
       
       wids = st.columns(4)
       feature_sels=[]
       for i, feature in enumerate(cols):
           with wids[i%4]:
              if st.checkbox(feature, value=False, key=f'chk_{feature}'):
                 feature_sels.append(feature) 
                 
       # st.write(feature_sels)
               
       st.dataframe(df_ai[feature_sels])
       
    if menu == menus[1]:   
       st.pyplot(ai.line_pro(df_ai)) 
        
        
    if menu == menus[2]:
        st.pyplot(ai.hist_pro(df_ai)) 
        
       
       
       
       
       
       
       
       
       
    
    
    
    
    
    
    
    

if __name__ == "__main__":
    
    # 현재 프로세스가 streamlit 환경 내에서 실행 중인지 확인
    if st.runtime.exists():
        main()
    else:
        # streamlit 환경이 아니면, 스스로를 streamlit으로 실행 (외부 터미널 효과)
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
        
        
        