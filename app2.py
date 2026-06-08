import streamlit as st
import project2.waferfunc as wf
import sys
import pandas as pd
from streamlit.web import cli as stcli
import numpy as np

# 맨 앞에 있어야함
st.set_page_config(page_title="letuin", layout="wide")


def main() :
    
    menus = ['EDA(데이터 탐색)', 'load_data', 'menu3']
    menu = st.sidebar.radio("project2 공정데이터 분석",menus)
    print(menu)
  
    if menu == menus[0]:
        st.markdown("##### EDA(데이터 탐색)") 
       
        df_pecvd, df_waf=wf.load_data()
        dfs=[df_pecvd, df_waf]
        df_option=['df_pecvd', 'df_waf']
        chk_df = st.radio("1. 데이터를 선택 하세요", df_option, 
                          horizontal=True, index=0)
        
        df_index= df_option.index(chk_df)
        
        st.dataframe(dfs[df_index])
       
        
    elif menu == menus[1]:
        st.markdown("##### heatmap 입니다")
        st.pyplot(wf.sample_load_data())
    elif menu == menus[2]:
        st.markdown("##### menu3 입니다")

if __name__ == '__main__':
    if st.runtime.exists():
        main()
    else:
        sys.argv = ["streamlit","run",sys.argv[0]]
        sys.exit(stcli.main())
        
   
# top bar : run --->     configuration per file --->
#           runner ---> external terminal 



    
        
        
        
        
        
        
        