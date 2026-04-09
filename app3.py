# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 11:00:10 2026

@author: letuin
"""
import streamlit as st
from streamlit.web import cli as stcli
import sys

st.set_page_config(page_title="대시보드", layout="wide")



def main():
    menu = st.sidebar.radio(
       "Project3 : 가상 계측(VM) 및 이상 감지(FDC)",
       [ "AI Data 검색",
        "📈 히스토그램",
        "📈 Line Chart",
        "📈 Target과 상관계수",      
        "분류 모델",  
        "회귀 모델"]
       )
    if menu == "AI Data 검색"  :
      st.title("AI Data 검색") 

if __name__ == "__main__":
    
    # 현재 프로세스가 streamlit 환경 내에서 실행 중인지 확인
    if st.runtime.exists():
        main()
    else:
        # streamlit 환경이 아니면, 스스로를 streamlit으로 실행 (외부 터미널 효과)
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
