import  streamlit as st 
# pip install streamlit
import  appService  as serv 
import sys 
from streamlit.web import cli as stcli
import cv2
from tensorflow.keras.models  import load_model
from PIL import Image
import numpy as np




# 1. 페이지 설정 
st.set_page_config(page_title="Streamlit", layout='wide')


def main():
    
    menus = ["1. getMenu", "2. calChoice", '3. Bar Chart', '4. 최대 과일량', '5. 손글씨 예측']
    
    menu = st.sidebar.radio("렛유인",menus)
    
    if menu == menus[0]: # getMean
        st.markdown(menu)
        lis = st.text_input('숫자로 1,2,3,4 입력 하세요 ') #문자열
        if st.button('제출'):
            lis= lis.strip(' ')  # 문자열
            lis= lis.split(',')  # 리스트
            lis = [ int(x)   for x in lis]
            textstr = serv.getMean(lis)
            st.markdown(f"### {textstr}")
            
            
        
        
        
        
        
        
    elif menu == menus[1]:
       
        st.markdown(menu)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('##### 연산자')
            op = st.radio("", ['+','-','*','/'], index=0, 
                          horizontal=True)
            sop = str(op)
        with col2:
            st.markdown('##### 초기값')
            first = st.number_input("초기값", 
                                    min_value=0,
                                    max_value=120,
                                    step=1,                                    
                                    value=20)
            
        with col3:
            st.markdown('##### 리스트')
            lis=st.text_input('숫자로 1,2,3,4 입력 하세요 ') #문자열
        
        if st.button('제출'):
            lis= lis.strip(' ')  # 문자열
            lis= lis.split(',')  # 리스트
            lis = [ int(x)   for x in lis]
            textstr = serv.calChoice(sop, first, *lis)
            st.markdown(f"### {textstr}")
    
    elif menu == menus[2]:
        st.markdown(menu)
        st.pyplot(serv.bar_pro())
        
        
    elif menu == menus[3]:  
        st.markdown(menu)
        dics = {'apple': 10, 'banana': 25, 'cherry': 25, 'orange':15}
        textstr = serv.maxValue(dics)
        st.markdown(f"### {textstr}")
   
    
    
    
    
if __name__ == "__main__": # 전체(external terminal) 실행을 했나?
     if st.runtime.exists():
         main()
     else:
         sys.argv = ['streamlit', "run", sys.argv[0]]
         sys.exit(stcli.main())
                            
                        
