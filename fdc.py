
def send_discord(msg):
    

    webhook_url = "https://discord.com/api/webhooks/1525755958708670624/yC4rrqVdDZP8K6DuUdrjkFnwWRohDl38reTerV3xJcdw_fIL1Fql9xFb1xkiQIix4aCO"
    data = {
        "content": msg,
       
    }
    
    response = requests.post(webhook_url, json=data)
    
    if response.status_code == 204:
        print("메시지 전송 성공!")
    else:
        print(f"실패 코드: {response.status_code}")


    =============================================
    
    elif menu == menus[10]:
        t_recipe=df_recipe.reset_index()
        st.markdown('##### '+ menu)
        st.write(st.session_state.to_dict())
        # --- 변수 및 세션 상태 초기화  ---
        if "run" not in st.session_state:
            st.session_state.run = False                   # 시계열 차트 실행
        if "current_index" not in st.session_state:
            st.session_state.current_index = 0    # .current_index > 0  기본 차트 실행
        if "befor_waf" not in st.session_state:   # 현재 실행 wafer name
            st.session_state.befor_waf = None
        if "disabled" not in st.session_state:          
            st.session_state.disabled = True  # 이상치 button disabled
        if "chk_point" not in st.session_state:          
            st.session_state.chk_point = False # 이상치 확인
        if "stop_name" not in st.session_state:          
            st.session_state.stop_name = 'Stop'  # button name
        
       
        
       
         
            
            
            
        # 전체 자료를 읽어 가면서 웨이퍼 Id를 확인하는데 해당 피처의 6시그마를 벗어 나면 error message를 popup한다 
        feat = st.radio("공정 자료를 선택하세요:", t_recipe.columns[7:13], horizontal=True,index=3)     
        # 시작 버튼 및 속도 조절 슬라이더
        speed = st.slider("데이터 생성 속도 (초)", min_value=0.1, max_value=1.0, value=0.001, step=0.1)        
        # button         
        col1, col2, col3 = st.columns([1, 1, 3])
        
        with col1:
            start_button = st.button("차트 그리기 시작")
            if start_button :
                st.session_state.run=True
                st.session_state.current_index=0
            
                
                st.session_state.disabled = True
                st.session_state.stop_name == 'Stop'
                st.rerun()
        with col2:
            
            start_button = st.button(st.session_state.stop_name ,  type="secondary" )
            
            if start_button:
              
                
                if st.session_state.stop_name == 'Stop':
                    st.session_state.stop_name="Restart"
                    st.session_state.run = False
                else :
                    st.session_state.stop_name='Stop'
                    st.session_state.run = True
                st.rerun()
                
       
        with col3:
            
            start_button = st.button("이상치 발생" , disabled=st.session_state.disabled , type="primary" )
            
            if start_button:
                st.session_state.disabled = True
                st.session_state.run = True
                st.rerun()
                
       
     
                           
        chart_placeholder = st.empty()           
        feat_list=t_recipe[feat].to_list()
        # 💡 해결책 1: 루프를 돌기 전에 미리 Y축의 최소값과 최대값을 구해놓습니다.
        feat_min = t_recipe[feat].min()
        feat_max = t_recipe[feat].max()
        feat_mean = t_recipe[feat].mean()
        feat_std = t_recipe[feat].std()
        
        ucl=feat_mean+(3*feat_std)
        lcl=feat_mean-(3*feat_std)
        # 2. Matplotlib Figure 객체 하나 생성
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.clear()
            
            
        wafs =t_recipe['WaferNo'].to_list() 
        xlen=len(feat_list)
       # 1. 차트를 그릴 empty 컨테이너 생성
        chart_placeholder = st.empty()
        ax.axhline(y=feat_mean, color='y', linestyle='--', linewidth=1.5)  
        ax.axhline(y=ucl, color='r', linestyle='--', linewidth=1.5) 
        ax.axhline(y=lcl, color='r', linestyle='--', linewidth=1.5) 
        
        wno_index = t_recipe.groupby('WaferNo')[feat]
        xtick_pos = []
        xtick_labels = []
        for wfname, groupdf in wno_index:
            last_idx = groupdf.index[-1]
            first_idx = groupdf.index[0]
            # print(wfname,  first_idx, last_idx)
            ax.axvline(x=last_idx, color='grey', linestyle='--', linewidth=1.5) 
            xtick_pos.append(last_idx)
            xtick_labels.append(t_recipe.loc[last_idx, 'WaferNo']) 
            
        # df_recipe
        ax.set_xlabel("시간 (Index)")
        ax.set_ylabel("값 (Value)")
        ax.set_xlim(0, xlen + 5) 
                  # X축 여유 공간 확보
        ax.set_ylim(feat_min - (feat_std*0.1), feat_max + (feat_std*0.1))       # Y축 여유 공간 확보
        ax.grid(True, linestyle='--', alpha=0.5)      # 격자 추가 (선택사항) 
        ax.set_xticks(xtick_pos)          # 눈금 위치 = last_idx
        ax.set_xticklabels(xtick_labels, rotation=45, ha='right')  # 눈금 라벨 = WaferNo                   
       
        if st.session_state.current_index > 0:
            x_data = list(range(st.session_state.current_index,))       # [0, 1, ..., i]
            y_data = feat_list[:st.session_state.current_index] 
            ax.set_title(wafs[st.session_state.current_index - 1]) 
            ax.plot(x_data, y_data, color="#1f77b4", linewidth=2, marker='o', mfc='#ff7f0e', mec='#ff7f0e', markersize=6)
            
            chart_placeholder.pyplot(fig)                   
        
                  
        
       # 2. 실시간 차트를 보여줄 빈 컨테이너 생성
        if st.session_state.run :    # 시각화 시작 멈출때는 False 를 한다
           
            for  i  in  range(st.session_state.current_index,xlen):
                st.session_state.current_index=i 
                x_data = list(range(i + 1))       # [0, 1, ..., i]
                y_data = feat_list[:i + 1]   
              # 그래프 그리기 (선 + 마커 고정)
                ax.plot(x_data, y_data, color="#1f77b4", linewidth=2, marker='o', mfc='#ff7f0e', mec='#ff7f0e', markersize=6)
                # 차트 레이아웃 최적화 (축 범위, 타이틀 설정)
                chart_placeholder.pyplot(fig)
                # 6 시그마 초과 이상치인경우 alert창 뛰우고 다음 wafer될때까지는 뛰우지 않는다 
                #### !!!해야 한다
                last_value = y_data[-1]  
                is_outlier = (last_value > ucl) or (last_value < lcl)
                is_new_wafer_for_alert = wafs[i] != st.session_state.befor_waf
                
                if st.session_state.chk_point:
                    send_discord(f'{datetime.now()}:{wafs[i]}에 {feat}가  {last_value} 입니다 ')
                    st.session_state.chk_point=False
                    st.session_state.run = False        # 잠깐 멈춤       
                    # st.session_state.current_index=i   # 기억자료 순번
                    st.session_state.befor_waf = wafs[i]  # 경고를 띄웠으므로 현재 웨이퍼 저장 (다음 루프에서 중복 경고 방지)   
                    st.session_state.disabled=False   #button 활성화
                    st.rerun()   # session은 그대로이고 패이지만 top부타 실행한다 
                
                
                if is_outlier and is_new_wafer_for_alert:
                    st.session_state.chk_point=True  # 치상치 차트를 확인 
                    
                    
                            
                    
                    
                ax.set_title(wafs[i])
               
                
                # 💡 핵심 2: st.pyplot은 중복 key 에러가 없으므로 그냥 fig를 넣어주면 됩니다!
                
    
                # 지정한 시간만큼 대기
                time.sleep(speed)
                
        st.success("데이터 시뮬레이션이 완료되었습니다!")
    
    
