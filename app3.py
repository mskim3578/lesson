# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 11:00:10 2026

@author: letuin
"""
import streamlit as st
from streamlit.web import cli as stcli
import sys
import project3.aifunc as ai
import numpy as np
import pandas as pd

st.set_page_config(page_title="대시보드", layout="wide")

@st.cache_data
def get_cache_dataset():    
    df_ai=ai.load_csv()
    return df_ai

df_ai=get_cache_dataset()


def main():
    
    menu = st.sidebar.radio(
       "Project3 : 가상 계측(VM) 및 이상 감지(FDC)",
       [ "AI Data 검색",
        "📈 히스토그램",
        "📈 Line Chart",
        "📈 Target과 상관계수",      
        "분류 모델",  
        "유/불량 확인",  
        "회귀 모델",
        "예상 target 값"]
       )
    if menu == "AI Data 검색"  :
      st.subheader("AI Data 검색") 
      
      st.dataframe(df_ai)
     
      search_char=st.text_input("1.찾는 문자를 입력하세요 :", placeholder="예)setpoint, CTC_chA_WaferNo=='1' ")       
      words=df_ai.columns
      cols=[w for w in words if search_char.lower() in w.lower()]    
      
      #   unique()
      chk_col = st.radio("2. 컬럼를 선택하세요:", cols, horizontal=True,index=0)
      
      temp_li=df_ai[chk_col].unique().tolist()   
             
      if len(temp_li) > 20 :
        result = f'{temp_li[:20]}... {len(temp_li)}개'
      else :
        result=f'{temp_li}  {len(temp_li)}개'
        
      st.write(result)
      temp_type= df_ai[chk_col].dtype
      if (temp_type == 'int64') or (temp_type == 'float64') :
            df=df_ai.select_dtypes(include=[np.number])   
            temp_df=pd.DataFrame([[]])
            temp_df['mean']=df[chk_col].mean()
            temp_df['min']=df[chk_col].min()
            temp_df['max']=df[chk_col].max()
            temp_df['std']=df[chk_col].std()
            st.dataframe(temp_df, use_container_width=True)
      
    elif menu == "📈 히스토그램"  :
      
      st.title("📈 히스토그램") 
      st.pyplot(ai.hist_pro(df_ai))
      
    elif menu == "📈 Line Chart"  :
      st.title("📈 Line Chart") 
      st.pyplot(ai.plot_pro(df_ai))
      
    elif menu == "📈 Target과 상관계수"  :
      target = st.radio("Target을 선택 하세요:", ['Depo_THK','Particle'],
                                    horizontal=True,index=0)
      st.title("📈 Target과 상관계수")       
      fig=ai.corr_pro(df_ai, target, 0.1)
      st.pyplot(fig) 
      
      
    elif menu == "분류 모델"  :
      st.title("분류 모델")   
      target = st.radio("Target을 선택 하세요:", ['Depo_THK','Particle'], 
                        horizontal=True,index=0)
      results, fig = ai.binary_model(df_ai, target)
      st.dataframe(results, use_container_width=True)
      st.pyplot(fig)
    
    elif menu ==  "유/불량 확인"   :
      st.title("유/불량 확인")   
      target = st.radio("Target을 선택 하세요:", ['Depo_THK','Particle'], 
                        horizontal=True,index=0)  
      df_cm, df_anormal=ai.binary_predict(df_ai, target)      
      st.dataframe(df_cm) 
      st.dataframe(df_anormal)  
        
        
        
    elif menu == "회귀 모델"  :
        st.title("회귀 모델")          
        target = st.radio("Target을 선택 하세요:", ['Depo_THK','Particle'], 
                          horizontal=True,index=0)  
        df_results, fig=ai.linear_model(df_ai, target)      
        st.dataframe(df_results) 
        st.pyplot(fig)
        
    elif menu == "예상 target 값":          
        target = st.radio("Target을 선택 하세요:", ['Depo_THK','Particle'], horizontal=True,index=0)
        df_linear=ai.linear_predict(df_ai, target)        
        st.dataframe(df_linear)
         

if __name__ == "__main__":
    
    # 현재 프로세스가 streamlit 환경 내에서 실행 중인지 확인
    if st.runtime.exists():
        main()
    else:
        # streamlit 환경이 아니면, 스스로를 streamlit으로 실행 (외부 터미널 효과)
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
