
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
    
