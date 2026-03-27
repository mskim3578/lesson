import streamlit as st
import project2.letuinfunc  as wf

# !streamlit run app.py

# 1. 페이지 설정 (웹 브라우저 탭 이름 및 아이콘)
st.set_page_config(page_title="Streamlit 대시보드", layout="wide")

# 처음 한 번만 로드되고, 이후 사용자가 위젯을 조작해도 다시 로드하지 않습니다.

@st.cache_data
def get_cached_data():  
    # 실제 데이터 로드 함수 호출
    df_pecvd, df_waf = wf.load_data()
    return df_pecvd, df_waf

df_pecvd, df_waf =get_cached_data()


# 2. 좌측 사이드바 메뉴 생성
with st.sidebar:
    st.title("📌 메인 메뉴")
    # 네 가지 메뉴 선택 라디오 버튼
    choice = st.radio(
        "이동할 페이지를 선택하세요",
        ["홈 (Home)", "데이터 분석", "머신러닝 모델", "설정", "Menu1", 'Menu2']
    )
    st.info("메뉴를 클릭하면 우측 화면이 변경됩니다.")

# 3. 우측 메인 화면 구성 (선택된 메뉴에 따라 출력)
if choice == "홈 (Home)":
    st.header("🏠 홈 화면")
    st.write("서비스의 메인 페이지입니다. 대시보드 개요를 확인하세요.")
    
    st.dataframe(df_waf, use_container_width=True)
    st.dataframe(df_pecvd, use_container_width=True)
    
    
    st.metric(label="오늘의 방문자", value="1,234명", delta="12%")

elif choice == "데이터 분석":
    st.header("📊 데이터 분석")
    st.subheader("반도체 웨이퍼 데이터 시각화")
    st.write("이곳에서 데이터 전처리 및 시각화 결과를 확인할 수 있습니다.")
    # 예시 차트
    import pandas as pd
    import numpy as np
    chart_data = pd.DataFrame(np.random.randn(20, 3), columns=['A', 'B', 'C'])
    st.line_chart(chart_data)

elif choice == "머신러닝 모델":
    st.header("🤖 머신러닝 모델")
    st.write("학습된 모델의 성능 지표와 예측 결과를 제공합니다.")
   
    st.pyplot(wf.sample_scatter())
    
    col1, col2 = st.columns(2)
    col1.success("모델 정확도: 98%")
    col2.warning("모델 손실률: 0.02")

elif choice == "설정":
    st.header("⚙️ 설정")
    st.write("사용자 계정 및 시스템 환경 설정을 변경할 수 있습니다.")
    st.toggle("다크 모드 활성화")
    st.text_input("사용자 이름", value="관리자")
