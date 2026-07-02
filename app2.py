import streamlit as st
import project2.waferfunc  as wf
import sys
import pandas as pd
from streamlit.web import cli as stcli
import numpy as np
import matplotlib.pyplot as plt 
plt.rcParams['font.family'] = 'Malgun Gothic'   # Windows
# 마이너스 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False
st.set_page_config(page_title="렛유인 공정 데이터 분석", layout="wide")

# 저장 폴더 설정
# SAVE_DIR = "uploads"
# if not os.path.exists(SAVE_DIR):
#     os.makedirs(SAVE_DIR)

# 페이지 설정
@st.cache_data
def get_all_dataset():
    """모든 외부 파일을 읽고 결합 및 전처리를 수행하여 리턴합니다."""
    
    with st.spinner('데이터 세트를 구성 중입니다...'):
        df_pecvd, df_waf, df_recipe=wf.load_data()
        print(len(df_pecvd), len(df_waf),len(df_recipe))
    st.write(f"PECVD: {len(df_pecvd)}건 | WAF: {len(df_waf)}건 | Recipe: {len(df_recipe)}건")
    return df_pecvd, df_waf, df_recipe  


# --- 메인 앱 실행부 ---


df_pecvd, df_waf, df_recipe = get_all_dataset()



def main():
    # 여기에 본인의 Streamlit 코드를 작성하거나 함수를 호출하세요.
    # 파일 경로 정의
    PATH_PECVD = "data/pjt2_eqp_paramters.xlsx"
    PATH_COORD = "data/pjt2_coordinate.xlsx"
    PATH_THK = "data/pjt2_thickness.xlsx"

   
   
    # 사이드바 메뉴
    
    st.sidebar.image('data/letuin.png')
    
    menus=  ["1.EDA (데이터 탐색)",   # 0
             "2.웨이퍼 좌표",         # 1
             "3.웨이퍼 Heatmap",     # 2
             "4.분산분석(ANOVA, Analysis of Variance)",      # 3
             "5.공정능력지수:CPK",  # 4
             "6.레시피상관계수",  #5
             "7.최소제곱법:OLS",    #6
             "8.시계열 자료",     #7
             "9.웨이퍼 불량패턴",    #8
             "10.thickness 최적레시피"]    #10
    
    menu = st.sidebar.radio(
        "Project2: 공정 데이터 분석",
        menus
      
    )
    
  
   
    if menu == menus[0]:
        st.markdown(menu)
       
        dfs = [df_pecvd, df_waf, df_recipe]
        df_option=['df_pecvd', 'df_waf', 'df_recipe']
        chk_df = st.radio("1. 데이터를 선택하세요:",df_option, horizontal=True,index=0)
        # 선택된 값의 인덱스(순서) 구하기
        df_index = df_option.index(chk_df)  # 선택된 df 이름으로 된 파일 위치
        
        
        search_char=st.text_input("2.찾는 문자를 입력하세요 :", placeholder="예)setpoint, CTC_chA_WaferNo=='1' ")        
        df=dfs[df_index]
        words=df.columns
        
        st.dataframe(df)
        # st.write(words)
        cols=[w for w in words if search_char.lower() in w.lower()]        
             
        chk_col = st.radio("3. 컬럼를 선택하세요:", cols, horizontal=True, index=0)
       
        temp_type= df[chk_col].dtype
        # st.write(df[chk_col].head())
        temp_df=pd.DataFrame([[]])
        if (temp_type == 'int64') or (temp_type == 'float64') :  
           
            temp_df['mean']=df[chk_col].mean()
            temp_df['min']=df[chk_col].min()
            temp_df['max']=df[chk_col].max()
            temp_df['std']=df[chk_col].std()
            temp_df['count']=df[chk_col].count()
            temp_df['unique']=len(df[chk_col].unique())
            
        st.dataframe(temp_df, use_container_width=True)
        #   unique()
        temp_li=df[chk_col].unique().tolist()
        if len(temp_li) > 10 :
            result = f'unique() : {temp_li[:10]}...[{len(temp_li)}]'
        else :
            result=f'unique() : {temp_li}' 
    
        st.write(result)
        
        if (len(search_char.strip()) > 0) :
            #st.dataframe(df[cols])
            
            
            st.markdown('#####  $ \sigma $ 기준  이상치(Line Chart) , 값의 평균 및 최빈값 이나 분포의 모양(Hist Chart)')
            
            # 1) EDA를 위한 선그래프를 그린다 
            st.pyplot(wf.all_line_chart(df, cols))                
          
            # 2) EDA를 위한 히스토그램를 그린다 
            st.pyplot(wf.all_hist_chart(df, cols))  
            st.pyplot(wf.skew_kurt_pro(df, cols))   
    # 라인 그래프 페이지
    
    elif menu == menus[1]:
       
        st.markdown("##### "+menu)      
        st.pyplot(wf.sample_scatter(df_waf))
       
    elif menu == menus[2]:
        
        st.markdown("##### "+menu)    
       
        cols=df_waf.columns[3:]
        selected_list = st.multiselect("웨이퍼를 선택하세요: 선택 내용이 없을시에는 모두 프린트 합니다 ", cols)
        chk_thk = st.radio("thickness value(y축) 고정 :", [True,False], horizontal=True,  index=1)
        
        
        if st.button("함수 실행"): 
            if len(selected_list) == 0 : selected_list=cols
            
            if chk_thk :
                st.pyplot(wf.wafer_chart_2(df_waf, selected_list))
            else:
                st.pyplot(wf.wafer_chart_1(df_waf, selected_list))
            
            
   
    elif menu == menus[3]:
       
        st.markdown("##### "+ menu)   
        st.pyplot(wf.box_pro(df_waf))
        
        st.markdown("##### 세 개 이상의 집단 간 평균의 차이가 통계적으로 유의미한지 확인하는 검정 방법")  
        
        target_col = st.radio("Target을 선택 하세요:", df_waf.columns[3:], horizontal=True, index=0)
        fig, df_sorted=wf.anova_pro1(df_waf, target_col)
        st.pyplot(fig)
        st.markdown('###### 분산분석(prerun기준)  :  p-value 순 Sort') 
        st.dataframe(df_sorted)
        st.markdown('###### 모든 recipe별   :  p-value') 
        df_f_val, df_p_val, fig = wf.nn_anova_pro(df_waf)
        st.pyplot(fig)
        st.markdown('###### 모든 recipe별   :  p-value 값') 
        st.dataframe(df_p_val)
        st.markdown('###### 모든 recipe별   :  f-value 값') 
        st.dataframe(df_f_val)
        
        
        
        
        
        # for ano in anovali:            
        #     fp = f'{ano[2]:<20s}  {ano[0]:<20.4f}  {ano[1]:<20.4f}'
        #     st.code(fp)
        
    
  
              
  
    
    
   
    
       
       
    elif menu == menus[4]:
       st.markdown("##### CPK(공정능력지수, Process Capability Index)")  
       st.markdown("###### 단순히 수율이 좋다 나쁘다를 넘어,  앞으로 불량이 날 가능성이 얼마나 되는가? ")
       
       target = st.radio("target :", ['mean',  'cmean'],  horizontal=True, index=0)
       
       # --- 2. USL을 조정하는 슬라이더 생성 ---
       # usl 구간적용 방법
       # 1  maan ~  max
       # 2  k * std 방법중 선택한다
       # st.slider(라벨, 최소, 최대, 기본값, 간격)
       
       
       
       
      
       sigma = st.slider(
           "규격 상한값(USL) 설정", 
           min_value=1.0,
           max_value=6.0,
           value=4.0,
           step=0.1
       ) 
    
       # k_sigma = float(st.text_input(label=f"추가 {sigma}", value=sigma))
       if target == 'cmean':
           df=df_waf.iloc[:, 3:]
       else:
           df=df_waf.iloc[:25, 3:]
           
       fig, df_sort, title=wf.cpk_pro(df,sigma)      
       
     
       # if st.button("함수 실행"): 
           
       # 2. 행 전체 스타일 정의 함수 (시리즈를 입력받음)
       def color_high_defect(row):
            
            if row['Cpk'] > 1.0: # 앞서 주신 코드의 기준(val > 1)에 맞게 조건을 수정하여 쓰세요!
                # 행의 모든 컬럼 개수만큼 동일한 스타일을 리스트로 반환 (행 전체 적용)
                return ['background-color: #f8d7da; color: blue; font-weight: bold'] * len(row)
            else:
                # 조건에 맞지 않으면 스타일 적용 안 함 (빈 문자열)
                return [''] * len(row)
        
        # 3. style.apply 적용 (axis=1 필수)
        # 이제 subset 지정을 빼고 전체 데이터프레임에 매핑합니다.
       styled_df = df_sort.style.apply(color_high_defect, axis=1)    
           
           
           
           
       # # 스타일 정의 함수
       # def color_high_defect(val):
       #    if val > 1:
       #      # 바탕색(background-color)과 글자색(color)을 함께 지정 가능
       #      return 'background-color: #d4edda; color: red; font-weight: bold'
       

       #  # 스타일 적용
       #  # subset을 사용하면 특정 컬럼에만 적용됩니다.  
       # styled_df = df_sort.style.map(color_high_defect, subset=['Cpk'])
       
       st.markdown(f"#### {title}")
       st.pyplot(fig)
       st.write("체크포인트")
       st.dataframe(styled_df)
        
    elif menu == menus[5]:
        st.markdown("##### "+menu)  
        # cols=['mean', 'uniformity']
        # chk_wafer = st.radio("target을 선택하세요:", cols, horizontal=True,
        # index=0)
        cols=df_recipe.columns[1:7].tolist() + ['mean','uniformity']
        st.pyplot(wf.recipe_coef(df_recipe[cols]))
        
        
    elif menu == menus[6]:
        st.markdown("##### OLS(Ordinary Least Squares, 최소제곱법)")  
        st.markdown("##### 실제 데이터 지점들과 모델 직선 사이의 거리(오차)의 제곱을 모두 더했을 때, 그 값이 최소가 되는 직선을 찾는 방법")
        sel=['mean', 'uniformity']
        chk_wafer = st.radio("target을 선택하세요:", sel, horizontal=True,index=0)
        
        
        
        
        
        cols = df_recipe.columns[1:7].tolist()+['mean','uniformity']
        fig, results  = wf.OLS_pro(df_recipe[cols], chk_wafer)
        st.pyplot(fig)
        
        
        # statsmodels 결과를 데이터프레임으로 변환
        coeff_df = pd.DataFrame({
            "Coefficient": results.params,
            "Std. Error": results.bse,
            "t-value": results.tvalues,
            "P-value": results.pvalues
        })
        
        # P-value가 0.05 미만인 행에 하이라이트 주기
        def highlight_significant(s):
            # return ['background-color: #d4edda' if s.name == 'P-value' and v < 0.05 else '' for v in s]
            return ['color: red' if s.name == 'P-value' and v > 0.05 else '' for v in s]
    
        st.dataframe(coeff_df.style.format("{:.4f}").apply(highlight_significant, axis=0), 
                     use_container_width=True)
    
    
    elif menu == menus[7]:
        
        wf_no = st.radio("recipe group을 선택하세요:", ['1,2,18' ,'3,4,5','6,7,8','9,10,11','12,13,14','15,16,17'], 
                                               horizontal=True,index=0)  
        prof = st.radio("자료를 선택하세요:", ['wafer/setpoint', 'wafer/flow', 'setpoint/flow', 'flow/wafer(wafer_no 선택 하세요)'], 
                                               horizontal=True,index=3)        
        st.pyplot(wf.timeserise_pro(prof, df_recipe, wf_no))
        
    elif menu == menus[8]:
       
          st.markdown("##### "+menus[8])    
          cols=df_waf.columns[3:]
          chk_wafer = st.radio("웨이퍼를 선택하세요:", cols, horizontal=True,
          index=1)
          fig, report_lines=wf.wafer_pattern(df_waf,chk_wafer)
          st.pyplot(fig)
          for s in report_lines:
                 st.write(s) 
            
        
    elif menu == menus[9]:    # 최적의 조건
        st.markdown('##### 평균 기준 최적 레시피 with uniformity')     
        
        cols=df_recipe.columns[1:7].to_list()+['mean','uniformity']
        
        X = df_recipe[cols]  
       
        t_mean = 3300.0
        t_uni = 5.0
        
        t_mean= st.slider(
            'mean',            
            max_value=t_mean*1.1,
            min_value=t_mean*0.9,    
            value=t_mean,
            step=0.1
        ) 
        
        t_uni= st.slider(
            'uniformity',            
            max_value=t_uni*2,
            min_value=t_uni/2,    
            value=t_uni,
            step=0.1
        ) 
        
        t_pro= st.slider(     # 중요도 %
            '비율(mean:uniformity)',            
            max_value=100,
            min_value=0,    
            value=50,
            step=5
        ) 
      
        w_mean= t_pro
        w_uni = (100-t_pro)
        st.markdown(f"##### w_mean : {w_mean}%   w-uni : {w_uni}%")
        # if y_mean != 0 :
        df_result= wf.set_recipe(X, t_mean, t_uni,w_mean,w_uni )        
        st.dataframe(df_result)
        
           
                    
               
            
            
            
    # elif menu == menus[10]:
    #     st.markdown('##### 균일성 추가  최적 레시피')
    #     tmean = int(st.text_input(label="mean", value=3300))      
    #     tuniformity = float(st.text_input(label="uniformity", value=0.005))
        
    #     df_out = wf.set_recipe2(df_recipe, "mean", "uniformity", 3300, 0.0305 )
        
    #     st.dataframe(df_out)
        
        
    elif menu == menus[11]:
        st.markdown('##### 레시피 재현성')
        tmean = int(st.text_input(label="mean", value=3300))
        tuniformity = float(st.text_input(label="uniformity", value=0.00))    
        
        
    # elif menu == "레시피별 웨이퍼 머신러닝":
    #     fig, li=wf.wafer_multi_model(df_wafer_setpoint, 7)
    #     st.pyplot(fig)
    #     for s in li:
    #         st.code(s)
        
        
        
    elif menu == "ℹ️ 정보":
        st.markdown("##### ℹ️ 정보")
        st.write("**버전:** 1.0.0")
        st.write("**제작:** Streamlit Dashboard")
        st.write("좌측 사이드바에서 메뉴를 선택하여 다양한 기능을 사용하세요.")
    


#   streamlit run app.py

if __name__ == "__main__":
    
    # 현재 프로세스가 streamlit 환경 내에서 실행 중인지 확인
    if st.runtime.exists():
        main()
    else:
        # streamlit 환경이 아니면, 스스로를 streamlit으로 실행 (외부 터미널 효과)
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())