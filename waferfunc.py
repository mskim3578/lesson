import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt 
import numpy as np
from scipy import stats
import seaborn as sns
from matplotlib.patches import Circle
pd.options.mode.chained_assignment = None
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.metrics import  confusion_matrix


from sklearn.preprocessing import OneHotEncoder


from scipy.optimize import   minimize
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

plt.rcParams['font.family'] = 'Malgun Gothic'   # Windows
# 마이너스 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False
scaler = StandardScaler()



# %%   load_data :  df_pecvd, df_waf, df_recipe
PATH_PECVD = "data/pjt2_eqp_paramters.xlsx"
PATH_COORD = "data/pjt2_coordinate.xlsx"
PATH_THK = "data/pjt2_thickness.xlsx"

def load_data(stepname='Depo'):
    
    # 1. EQP Parameters 로드 및 전처리
    df_all = pd.read_excel(PATH_PECVD, engine='openpyxl')
    #1)  df_all['chA_VIR_Step_Name']=='Depo'인것만 가지고 온다 
    df_pecvd = df_all[df_all['chA_VIR_Step_Name']=='Depo']
    
    #2)  null이 있는 column을 삭제 한다 
    df_pecvd.dropna(axis=1, inplace=True)  #  
    
    #3)  df_pecvd["chA_AO_mfc2_setpoint_Si2H6"] != 0 
    df_pecvd = df_pecvd[df_pecvd["chA_AO_mfc2_setpoint_Si2H6"] != 0]
    
    #4) AUTO 2250 0    astype('int')임   astype(int) X 아님
    df_pecvd['chA_VIR_APC_Setpoint']=df_pecvd['chA_VIR_APC_Setpoint'].str.split(' ').str[1]  
    df_pecvd['chA_VIR_APC_Setpoint']=df_pecvd['chA_VIR_APC_Setpoint'].astype('int')
    df_pecvd.info()
   
    #5)  columns의 unique()가 1인것은 삭제한다 
    df_pecvd = df_pecvd.drop(columns=[c for c in df_pecvd.columns if df_pecvd[c].nunique() <= 1])    
    
    
    
    # 2. Coordinate & Thickness 결합
    df_coor = pd.read_excel(PATH_COORD, engine='openpyxl')
    df_thk = pd.read_excel(PATH_THK, sheet_name='thickness', engine='openpyxl')
    
    # 컬럼명 공백 제거 (' Run#1' -> 'Run#1')
    df_thk.columns = [s.strip() for s in df_thk.columns]
    
    # 좌표(df_coor)와 두께(df_thk의 첫 번째 컬럼 제외한 나머지) 결합
    df_waf = pd.concat([df_coor, df_thk.iloc[:, 1:]], axis=1)
    
    df_recipe= setpoint_pro(df_pecvd, df_waf)
    
    return df_pecvd, df_waf, df_recipe    


def setpoint_pro(df_pecvd, df_waf):   
   
    setpoint=[
     'chA_VIR_Recipe_Name', # Recipe_Name    
     'chA_AO_mfc2_setpoint_Si2H6', # Si2H6  
     'chA_AO_mfc3_setpoint_N2O',  # N2O O는 영문 대문자
     'chA_AO_mfc10_setpoint_TN2', #TN2
     'chA_VIR_APC_Setpoint',  # Pressure
     'USER RF SET',   #Power   기존 recipe랑 다름
     'chA_VIR_Heater_Temp_Set' ]  #Temp        
    
    flow = ['chA_AI_mfc2_flow_Si2H6', # Si2H6
           'chA_AI_mfc3_flow_N2O',   # N2O O는 영문 대문자
           'chA_AI_mfc10_flow_TN2', 
           'chA_AI_Manometer_Pressure',
           'FORWARD POWER',   #Power
           'chA_VIR_Heater_Temp'  #Temp  
           ]
    
    df_t=df_pecvd[setpoint+flow] 
    # 컬럼명 수정
    df_t.columns=['WaferNo', 'Si2H6', 'N2O', 'N2', 'Pressure', 'Power', 'Temp','flow_Si2H6', 'flow_N2O', 'flow_N2', 'flow_Pressure', 'flow_Power', 'flow_Temp']

    mean_list=[]
    unif_list=[]
    for w in  df_t['WaferNo'] :
        wno=w.split('_')[3]
        tdf = df_waf['Run#'+wno]
        tmean=tdf.mean()
        tuni= (tdf.max()-tdf.min())/(tdf.max() + tdf.min())*100
        mean_list.append(tmean)
        unif_list.append(tuni)
    df_t['mean']=mean_list
    df_t['uniformity']=unif_list     
   
    
    return df_t
    
''' 
#  #5) 수치형 피처만 최종 선택

df_pecvd, df_waf, df_recipe =load_data()
df_pecvd = df_pecvd.select_dtypes(include=['number']) 
df_recipe.groupby([WaferNo])
df_recipe.columns


df_pecvd.info()
df_pecvd['chA_VIR_APC_Setpoint'].unique()
a1.unique()
'''

# %%   

def  all_line_chart(df, df_col):   #######################################
        
     # AUTO 2250 0 2250만 남기기
     
     # df['chA_VIR_APC_Setpoint']=df['chA_VIR_APC_Setpoint'].str.replace(r'[^0-9]', '', regex=True).astype(int)
     # df['chA_VIR_APC_Setpoint']=df['chA_VIR_APC_Setpoint']//10
    
     df=df.select_dtypes(include='number')   # 숫자
     plt.rcParams['font.family'] = 'Malgun Gothic'   # Windows
     # 마이너스 깨짐 방지
     plt.rcParams['axes.unicode_minus'] = False
     
     fig = plt.figure(figsize=(12, 10),linewidth=0,          # 테두리 두께 0
                 edgecolor='white',    # 테두리 색상을 흰색으로
                 facecolor='white')    # 배경색 흰색 #(width, height)
     # cols = int(math.sqrt(len(df_col)) ) + 1
     cols=4
     rows = len(df_col)*2//cols + 1
     print(cols, rows)
     count = 1
     
     for col in df_col:
         
         #  line chart
         plt.subplot(rows, cols, count)
         count +=1
         one_plot_chart(df, plt, rows, cols, count, col)
         plt.subplot(rows, cols, count)
         count +=1
         one_hist_chart(df, plt, rows, cols, count, col) 
         
     plt.tight_layout()      
     
     plt.show()
     return plt 


def one_plot_chart(df, plt, rows, cols, count, col):
     plt.title(col, fontsize=10)
     tmean=df[col].mean()
     tstd=df[col].std()*3
     plt.axhline(y=(tmean+tstd), color='blue', linestyle='dashed',
                 linewidth=2, label=(tmean+tstd))
     plt.axhline(y=(tmean-tstd), color='r', linestyle='dashed', 
                 linewidth=2, label=(tmean-tstd))
     plt.axhline(y=tmean, color='y', linestyle='dashdot', 
                 linewidth=3, label=(tmean-tstd))
     
    
     plt.plot(range(len(df)) ,df[col])
     # plt.axis("off")  #x, y 좌표 프린트 않한다  
     # plt.legend()

def  all_hist_chart(df, df_col): 
     fig = plt.figure(figsize=(12, 10),linewidth=0,          # 테두리 두께 0
                edgecolor='white',    # 테두리 색상을 흰색으로
                facecolor='white')    # 배경색 흰색 #(width, height)
     cols=3
     rows = len(df_col)//cols + 1
     print(cols, rows)
     count = 1
    
     for col in df_col:     
         plt.subplot(rows, cols, count)
         plt.title(col)         
         plt.hist(df[col])
         count +=1
     plt.tight_layout()
     plt.show()
     return plt   

def skew_kurt_pro(df, cols):    
    fig = plt.figure(figsize=(20, 20),linewidth=0,          # 테두리 두께 0
               edgecolor='white',    # 테두리 색상을 흰색으로
               facecolor='white')    # 배경색 흰색 #(width, height)
    plt.rcParams['font.family'] = 'Malgun Gothic' # 한글 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False
    
    width=3
    rows = len(cols)//width + 1
    print(width, rows)  # width, height
    count = 1
    for col in cols:
        sr=df[col]
        # 2. 왜도 및 첨도 계산
        skew_val = sr.skew()
        kurt_val = sr.kurt()
        
        # 3. 차트 그리기
        plt.subplot(rows, width, count)
        count +=1
        
        # 히스토그램과 밀도 곡선(KDE)을 동시에 플로팅
        sns.histplot(data=sr, color='#2bc0d3', bins=30,  alpha=0.6)
        
        # 주요 통계 지표 기준선 표시 (평균 vs 중앙값)
        mean_val = sr.mean()
        median_val = sr.median()
        plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'평균 (Mean): {mean_val:.1f}')
        plt.axvline(median_val, color='green', linestyle='-', linewidth=2, label=f'중앙값 (Median): {median_val:.1f}')
        
        # 4. 차트 내부에 왜도/첨도 수치 박스 삽입
        textstr = '\n'.join((
            f'● 왜도 (Skewness): {skew_val:.3f}',
            f'● 첨도 (Kurtosis): {kurt_val:.3f}'
        ))
        
        # box 속성을 이용해 차트 우측 상단에 하얗게 텍스트 상자를 띄웁니다.
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
        plt.gca().text(0.65, 0.85, textstr, transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', bbox=props, fontweight='bold')
        
        # 타이틀 및 레이블 설정
        plt.title(f'■ {col}', fontsize=16, pad=15, fontweight='bold')
        plt.xlabel(f'{col}', fontsize=12)
        plt.ylabel('밀도 (Density)', fontsize=12)
        plt.legend(loc='upper left')
        plt.grid(True, linestyle=':', alpha=0.6)
        
    plt.tight_layout()  # 그래프 간격 자동 조정
    plt.show()
    return fig


def one_hist_chart(df, plt, rows, cols, count, col):
    sr=df[col]
    # 2. 왜도 및 첨도 계산
    skew_val = sr.skew()
    kurt_val = sr.kurt()  
    
    # 히스토그램과 밀도 곡선(KDE)을 동시에 플로팅
    plt.hist(sr, color='#2bc0d3', bins=30,  alpha=0.6)
    
    # 주요 통계 지표 기준선 표시 (평균 vs 중앙값)
    mean_val = sr.mean()
    median_val = sr.median()
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'평균 (Mean): {mean_val:.1f}')
    plt.axvline(median_val, color='green', linestyle='-', linewidth=2, label=f'중앙값 (Median): {median_val:.1f}')
    
    textstr=f'왜도(Skewness):{skew_val:.3f} \n첨도(Kurtosis):{kurt_val:.3f}'   
    # box 속성을 이용해 차트 우측 상단에 하얗게 텍스트 상자를 띄웁니다.
    # props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    # plt.gca().text(0.1, 0.1, textstr, transform=plt.gca().transAxes, fontsize=7,
    #         verticalalignment='top', bbox=props, fontweight='bold')
    
    # 타이틀 및 레이블 설정
    plt.title(textstr, fontsize=6)
    plt.legend(fontsize=6)
    plt.grid(True, linestyle=':', alpha=0.6)
    
 

# %%   test chart

def sample_scatter(df_waf):  
   
    # 웨이퍼 데이터 시각화
    fig=plt.figure(figsize=(10, 8))
    plt.scatter(df_waf["x"], df_waf["y"], c=df_waf['Run#1'], cmap='coolwarm', s=700)
    for idx, row in df_waf.iterrows():
        plt.text(
            row['x'],           # x 위치
            row['y'],           # y 위치
            str(idx+1),  # 표시할 값
            fontsize=12,
            ha='center',        # 수평 정렬
            va='center',        # 수직 정렬
            fontweight='bold',
            color='black'
        ) 
    plt.colorbar(label='Run#1 colorbar')
    plt.title('Wafer Measurement Locations and Values')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    # plt.show()    
    return fig

# sample_scatter()  

# %%
# 1. 가상의 웨이퍼 측정 데이터 설정 (x, y 좌표 및 두께 Thickness)
# 웨이퍼 중심을 (0,0)으로 가정 (단위: mm, 12인치 웨이퍼 기준 약 150mm 반경)


def wafer_chart_1(df_waf, collist):  
    
    xy_list=[]
    for x, y in zip(df_waf['x'], df_waf['y']):
        xy_list.append([x,y])
    points = np.array(xy_list)
    fig=plt.figure(figsize=(20, 20)) # 3x6 배열이므로 가로를 길게 설정

    for i, col in enumerate(collist):
        # 측정된 두께 데이터 (예: 가스 유량 문제로 Center가 두꺼운 패턴 가정)
        values = np.array(df_waf[col].to_list()) 
        plt.subplot(4, 5, i + 1) 
        # 2. 그리드 생성 (등고선을 그리기 위한 정밀한 그물망)
        grid_x, grid_y = np.mgrid[-150:150:100j, -150:150:100j]
        
        # 3. 데이터 보간 (측정되지 않은 지점의 값을 추정)
        grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')
        
       
      
        
        # 등고선 채우기 (Contourf)
        contour = plt.contourf(grid_x, grid_y, grid_z, levels=15, cmap='RdYlBu_r')
        #plt.colorbar(contour, label='Thickness (Å)')
        
        # 등고선 라인 추가
        lines = plt.contour(grid_x, grid_y, grid_z, levels=15, colors='black', linewidths=0.5)
        plt.clabel(lines, inline=True, fontsize=8)
        
        # 실제 측정 지점 표시
        plt.scatter(points[:, 0], points[:, 1], color='white', marker='.', label='Measurement Points')
        
        # vmin, vmax를 설정하면 여러 맵의 색상 기준을 통일하여 비교하기 좋습니다.
        
        
        # 웨이퍼 외형(원) 그리기
        wafer_outline = plt.Circle((0, 0), 150, color='gray', fill=False, linestyle='--')
        plt.gca().add_artist(wafer_outline)
        plt.xlim(2800, 4800)
        plt.title(f'{col} Wafer Thickness (PECVD)')
        plt.xlabel('X Position (mm)')
        plt.ylabel('Y Position (mm)')
        plt.axis('equal')
        plt.grid()
    plt.tight_layout()
    plt.legend()
    # plt.show()
    return fig


def wafer_chart_2(df_waf, collist):  
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import griddata

    radius = 150  
    grid_x, grid_y = np.mgrid[-radius:radius:150j, -radius:radius:150j]
    grid_mask = grid_x**2 + grid_y**2 <= radius**2
    
    # 4x5 레이아웃에 맞춰 figsize 조정 (가로가 긴 형태가 유리)
    fig, axes = plt.subplots(4, 5, figsize=(22, 18))
    axes_flat = axes.flatten()

    img = None # 컬러바 기준 객체 초기화

    for i, col in enumerate(collist):
        if i >= 20: break  # 4x5=20개까지 출력
        
        ax = axes_flat[i]
        
        # 1. griddata 보간
        grid_z = griddata((df_waf["x"], df_waf["y"]), df_waf[col], (grid_x, grid_y), 
                          method='linear')
        
        # 2. 원 바깥쪽 마스킹
        grid_z[~grid_mask] = np.nan  
        
        # 3. Heatmap 출력 (imshow)
        # vmin, vmax를 고정하여 모든 웨이퍼의 색상 기준 통일
        img = ax.imshow(grid_z, extent=(-radius, radius, -radius, radius), 
                        origin='lower', cmap='coolwarm', vmin=2800, vmax=4000)
        
        ax.set_title(f'{col}', fontsize=12)
        ax.grid(False) # 웨이퍼 맵 특성상 격자선은 끄는 것이 깔끔합니다.
        # ax.axis('off') # 축 정보를 완전히 숨기려면 활성화

    # 사용하지 않는 나머지 서브플롯(빈 칸) 숨기기
    for j in range(i + 1, 20):
        axes_flat[j].axis('off')

    # 4. 상단에 공통 컬러바 추가
    # ax=axes를 주어 전체 서브플롯 영역 위에 배치합니다.
    if img is not None:
        cbar = fig.colorbar(img, ax=axes, location='top', shrink=0.4, pad=0.01, aspect=40)
        cbar.set_label('Measurement Value (Range: 2800 - 4000)', fontsize=14, labelpad=15)

    # layout 조정 (pad 값으로 간격 미세조정 가능)
    plt.subplots_adjust(top=0.70, hspace=0.3, wspace=0.3)
    
    plt.show()
    return fig

def wafer_chart_3(df_waf, collist):  
    # print(collist)
    radius = 150  # 웨이퍼 반지름 예시
    # 이 줄은 가로, 세로가 각각 -radius부터 +radius까지인 좌표 평면을 만듭니다.
    grid_x, grid_y = np.mgrid[-radius:radius:150j, -radius:radius:150j]

    # 숫자에 j가 붙으면 "150개를 생성하라"는 뜻입니다. 
    # 즉, 가로 150칸, 세로 150칸으로 촘촘하게 쪼갠 총 22,500개($150 \times 150$)의 좌표점을 생성합니다.
    grid_mask = grid_x**2 + grid_y**2 <= radius**2
    # grid_mask: 각 좌표점에 대해 이 식이 참(True)이면 원 안쪽, 거짓(False)이면 원 바깥쪽(사각형의 모서리 부분)으로 분류합니다.
    fig=plt.figure(figsize=(20, 20)) # 3x6 배열이므로 가로를 길게 설정
    
    # 분석할 컬럼 추출 (id와 마지막 컬럼 제외)
    

    for i, col in enumerate(collist):
        if i >= 19: break  # 3x6=18개까지만 출력 가능하도록 제한
        
        # 1. griddata를 사용하여 데이터 보간
        # df_coor에 각 포인트의 x, y 좌표가 있고 df_thk에 측정값이 있다고 가정
        # linear,nearest,cubic
        grid_z = griddata((df_waf["x"], df_waf["y"]), df_waf[col], (grid_x, grid_y), 
                                      method='linear')
        
        # 2. 원 바깥쪽 마스킹 처리 (NaN 설정)
        grid_z[~grid_mask] = np.nan  
        
        # 3. subplot 설정 (3행 6열, 인덱스는 1부터 시작)
        plt.subplot(4, 5, i + 1)
        
        # 4. Heatmap 출력
        # vmin, vmax를 설정하면 여러 맵의 색상 기준을 통일하여 비교하기 좋습니다.
        img = plt.imshow(grid_z, extent=(-radius, radius, -radius, radius), 
                         origin='lower', cmap='coolwarm',vmin=2800, vmax=4000)
        
       
        
        plt.title(f'{col}', fontsize=10)
    
        plt.grid(True)
        
     
        
        # plt.axis('off') # x, y 축 라벨이 겹치므로 깔끔하게 제거 (선택 사항)

    # 전체 레이아웃 조정 및 컬러바 하나로 통합하거나 개별 표시
    # plt.tight_layout()
    # 전체 화면 우측에 컬러바 하나만 크게 배치 (선택 사항)
    # plt.colorbar(img, ax=plt.gcf().get_axes(), label='Thickness (A)') 
    plt.colorbar(img, label='Measurement Value')
    plt.tight_layout()
    plt.show()
    return fig
  


# %%  anova df_waf를 기준으로 레시피의 제현성을 확인 한다 
# 의미 없음   
def anova_pro1(df_waf, target):
    anovali = []
    
    
     
    for col in df_waf.columns[3:]:
        f_val, p_val = stats.f_oneway(df_waf[col], df_waf[target])   # Prerun1
        anovali.append([p_val, f_val])
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    df_anova = pd.DataFrame(anovali)
    df_anova.index = df_waf.columns[3:].tolist()
    df_sorted = df_anova.sort_values(by=1)

    ax1 = df_sorted[1].plot(kind='bar', figsize=(12, 6), color='skyblue', edgecolor='black', label='F-통계량')
    ax1.grid(True, linestyle='--', alpha=0.6)

    for p in ax1.patches:
        height = p.get_height()
        ax1.annotate(f'{height:.2f}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='center',
                    xytext=(0, 7),
                    textcoords='offset points',
                    fontsize=10)

    ax2 = ax1.twinx()
    df_sorted[0].plot(kind='line', color='r', ax=ax2, label='p_val')        # ✅ ax2. 제거
    hline = ax2.axhline(y=0.05, color='orange', linestyle='--', linewidth=2, label='0.05 기준')  
    
    # ✅ ax2.axhline으로 변경
    # ✅ 3개 legend 합치기
    lines1, labels1 = ax1.get_legend_handles_labels()   # F-통계량
    lines2, labels2 = ax2.get_legend_handles_labels()   # p_val + 0.5 기준선

    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper center',
               fontsize=11,
               frameon=True,
               title='범례')
    # ax2.get_legend().remove()  # ✅ ax2 범례 제거 (중복 방지)

    plt.tight_layout()
    df_sorted.columns=['p_value', 'f_val']
    # plt.show() 
    return fig, df_sorted


# n by n anova
def nn_anova_pro(df_waf):
    # 1. 데이터를 저장할 빈 리스트 생성
    f_matrix = []
    p_matrix = []
    cols=df_waf.columns[3:]
    for col1 in cols:
       f_row = []
       p_row = []
    
       for col2 in cols:
            # 자기 자신과의 비교이거나 데이터에 결측치가 있을 수 있으므로 f_oneway 수행
            # (만약 두 집단의 데이터가 완전히 같다면 F값은 0, p-value는 1이 나옵니다)
            f_val, p_val = stats.f_oneway(df_waf[col1], df_waf[col2])
            # if p_val > 0.05 :
            #     p_val = 1   #다른 레시피
            # else:
            #     p_val = 0   #같은 레시피
            # 각각의 리스트에 계산된 값 추가
            f_row.append(f_val)
            p_row.append(p_val)
        
       # 한 행의 루프가 끝나면 전체 매트릭스 리스트에 추가
       f_matrix.append(f_row)
       p_matrix.append(p_row)

    # 3. 요청하신 구조(인덱스=col1, 컬럼=col2)로 DataFrame 생성
    df_f_val = pd.DataFrame(f_matrix, index=cols, columns=cols)
    df_p_val = pd.DataFrame(p_matrix, index=cols, columns=cols)
    
    
    # 2. 시각화 영역 생성
    fig, ax = plt.subplots(figsize=(10, 8))

    # 3. Seaborn Heatmap 그리기
    sns.heatmap(
        df_p_val, 
        annot=False,             # 각 칸에 F-통계량 수치 표시
        # fmt=".2f",              # 소수점 둘째 자리까지 제한
        cmap="Blues",         # 값이 클수록 밝아지는 직관적인 컬러맵 (또는 'rocket', 'magma')
        linewidths=0.8,         # 각 셀 사이의 경계선 두께
        linecolor="white",       # 경계선 색상
        square=True,            # 정사각형 격자 형태로 고정하여 가독성 향상
        ax=ax,
        cbar_kws={'label': 'F-Statistic Value'} # 컬러바 레이블 추가
    )
    
    return df_f_val, df_p_val, fig
#  시각화: Box Plot (상자 수염 그림)
def box_pro(df_waf):
    boxlist=[df_waf[col] for col in df_waf.columns[3:] ]
    fig=plt.figure(figsize=(15,7))  
    
    target_waf=df_waf['Prerun1']
    bp = plt.boxplot(boxlist, patch_artist=True)
    ucl=target_waf.mean()+target_waf.std()*3  # 3* 시그마
    lcl=target_waf.mean()-target_waf.std()*3
    t=bp['boxes']
    
    # bp에는 boxplot object가 리스트로 저장되어 있음
    cols =  df_waf.columns[3:]
    plt.xticks(range(1, len(cols) + 1),cols, rotation=45)
    ax = plt.gca() # 현재 활성화된 축(Axis)을 가져옵니다.
    labels = ax.get_xticklabels() # x축의 모든 글자 객체 리스트
    
    for i in range(len(cols)):
        if (df_waf[cols[i]].max() > ucl) | (df_waf[cols[i]].min() < lcl) :
            bp['boxes'][i].set_facecolor('orange')
            labels[i].set_color('red')       # 한계 돌파한 컬럼은 빨간색 글씨
            labels[i].set_weight('bold')
           
        else:
            bp['boxes'][i].set_facecolor('yellow')
            labels[i].set_color('black')       # 한계 돌파한 컬럼은 빨간색 글씨
            labels[i].set_weight('bold')
    
    plt.axhline(y=ucl , color = 'r' ,  label='ucl:'+str(ucl))
    plt.axhline(y=lcl , color = 'b' ,  label='lcl:'+str(lcl))
    plt.axhline(y=target_waf.mean() , color = 'y' ,  label='lcl:'+str(lcl))
    # plt.xticks(range(1, len(cols) + 1),cols, rotation=45)
    plt.xlabel("wafer Id")
    plt.ylabel("thickness std")
    # plt.title('wafer 분포표')
    
    #plt.legend()
    # plt.show()
    return fig




# def pecvd_anova_pro():
    

# %%
# 웨이퍼 레시피 와 웨이퍼의 평균, 표준편차, 균질성을 기준의 cpk를 연산 하게 한다  
# 각각의 값의 기준을 변동 하면서 cpk를 학인 하는 프로그램을 작성한다 


    
def cpk_pro(df, sigma): 
    
    ttarget=df.mean().mean()  #  전체 target기준
    tstd=df.std().mean()      # 표준 편차의 평균
    USL = ttarget + (tstd*sigma)
    LSL = ttarget - (tstd*sigma)
      
    
    print(sigma)  
    
    
    # 2. 개별 Run의 Cpk 계산
    ndf=pd.DataFrame([], index=df.columns[3:])
    ndf['wafer']=df.columns[3:]
    ndf['cpu'] = (USL - df.mean()) / (3 * df.std())
    ndf['cpl'] = (df.mean() - LSL) / (3 * df.std())
    # df['wafer']=df.index  
    # 각 행별로 더 작은 값을 Cpk로 결정
    ndf['Cpk'] = ndf[['cpu', 'cpl']].min(axis=1)
    
    # 3. 결과 할당 (모든 행에 동일한 Cpk 값을 넣거나 변수로 반환)
    
    # 1. Z-score 계산 (Cpk는 한쪽 규격 기준 최솟값이므로 3을 곱함)
    z_score = 3 * ndf['Cpk']
    
    # 2. 규격 안(양품)에 들어올 확률 계산 (Cumulative Distribution Function)
    yield_rate = stats.norm.cdf(z_score)
    
    # 3. 규격 밖(불량)으로 나갈 확률
    defect_rate = 1 - yield_rate
    
    # PPM (100만 개당 불량 수) 환산
    # ppm = defect_rate * 1_000_000
    
    
    ndf['불량률']=defect_rate
    
    df_sort=ndf.sort_values(by='Cpk', ascending=False)
    
    
    
    # uniformity는 edge는 제외하고 한다 ?
    # cp, cpk를 정한다 
    title=f'USL:{int(USL)}    LSL:{int(LSL)} '
    # box plot이 좋은듯
    fig=plt.figure(figsize=(10, 3)) # 3x6 배열이므로 가로를 길게 설정
  
    plt.bar(range(len(df_sort)),df_sort['Cpk'])
    plt.xticks(range(len(df_sort)), df_sort.index, rotation=45,      # 45도 회전
        fontsize=9,      # 글자 크기 조정 (원하는 수치로 변경 가능)
        ha='right')        # 회전 시 글자 끝처리를 눈금에 맞춤 (가독성 향상 핵심!)
    # 규격 한계 (Spec Limit) - 빨간색
    plt.axhline(y=1.33, color='r', linestyle='-', linewidth=2, label='1.33')
    plt.axhline(y=1.0, color='y', linestyle='-', linewidth=2, label='1.0')
    
    # # 관리 한계 (Control Limit) - 주황색/노란색 점선
    # plt.axhline(y=USL, color='orange', linestyle='--', linewidth=1.5, label='UCL (3350)')
    # plt.axhline(y=LSL, color='orange', linestyle='--', linewidth=1.5, label='LCL (3250)')
    
    # plt.show()
    plt.legend()
    return fig, df_sort, title
    
    
    
    
   


'''

# usl을 계산 하기 위한 계산룰을 정한다 

df_recipe=setpoint_pro(df_pecvd, df_waf)

df_pecvd, df_waf, df_recipe=load_data2()

cpk_pro(df_recipe, 'uniformity', 3.9)

df_recipe['mean'].mean()
df_recipe['mean'].max()
col49=['0', '1',
       '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14',
       '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26',
       '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38',
       '39', '40', '41', '42', '43', '44', '45', '46', '47', '48']

df_recipe[col49].mean().mean()

# 1. 공정 데이터 통계량 확인
avg_mean = df['mean'].mean()
avg_std = df['std'].mean()
avg_uni = df['uniformity'].mean()

df=pd.DataFrame([])
df['cpu'] = (USL - df_recipe['mean']) / (3 * df['std'])
df['cpl'] = (df['mean'] - LSL) / (3 * df['std'])

# 각 행별로 더 작은 값을 Cpk로 결정
df['Cpk'] = df[['cpu', 'cpl']].min(axis=1)




#####   st.sidebar.slider("Sigma 가중치 (k)", 1.0, 6.0, 3.0, step=0.1)

'''


# %%
def recipe_coef(df):   
    
    fig =  fig=plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='Blues')
    # plt.show()
    
    return fig
  
'''
df_recipe=setpoint_pro(df_pecvd, df_waf)
df=df_recipe[df_recipe.columns[:7]]
target='mean'
df[target] = df_recipe[target].values

fig=plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
# plt.show()




'''


# %%
def   OLS_pro(df, target):  
    
    plt.rcParams['font.family'] = 'Malgun Gothic'   # Windows
    # 마이너스 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False
    # 1. 실제값 vs 예측값 산점도     
    y = df[target]
    X = df.drop(target, axis=1)     
    # X 데이터 정규화
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    #df_wafer_setpoint.info()
    
    
    # 상수항을 추가하는 이유 r2 의 외곡을 방지
    X_final = sm.add_constant(X_scaled)    
    # 3. OLS 모델 학습
    model = sm.OLS(y, X_final)
    results = model.fit()
    y_pred = results.fittedvalues
    # 4. 결과 출력 (이미지와 동일한 형태의 표 출력)
    print(results.summary())
    print(X.columns)
    print(X.values)
    
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_final.columns
    vif_data["VIF"] = [variance_inflation_factor(X_final, i) for i in range(len(X_final.columns))]
    
    print(vif_data.sort_values(by="VIF", ascending=False))
    
    
    
   
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
   
   
    # axes[0,0].scatter(y,  results.fittedvalues)
    # axes[0,0].set_xlabel("Actual")
    # axes[0,0].set_ylabel("Predicted")
    # axes[0,0].set_title("1) Actual vs Predicted [모델이 얼마나 잘 맞췄나?]")
   
    
    # # 2. 잔차의 정규성 확인 (Q-Q Plot)
    # # 점들이 직선 위에 있을수록 모델이 신뢰할만합니다.
    
    # sm.qqplot(results.resid, line='s',ax=axes[0, 1])
    # axes[0,1].set_title("2) Normal Q-Q Plot [잔차의 정규성 확인]")
   
    
    
    # 4. 시각화 (2개의 그래프)   
    
    
    # 그래프 1: Actual vs Predicted (모델이 얼마나 잘 맞췄나?)
    
    axes[0].scatter(x=y, y=y_pred,  s=100, color='blue', edgecolor='black')
    axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2) # 45도 점선
    axes[0].set_title( f'3) 모델이 얼마나 잘 맞췄나? (R2: {results.rsquared:.3f})')
    axes[0].set_xlabel('Actual '+target)
    axes[0].set_ylabel('Predicted ' + target)
    
    # 그래프 2: Feature Importance (어떤 인자가 가장 중요한가?)
    # 상수항(const)을 제외한 계수(coef) 시각화
   
    coef_df = results.params.drop('const').sort_values()
    coef_df.plot(kind='barh',  color='skyblue', ax=axes[1])
    axes[1].set_title('4) 어떤 인자가 가장 중요한가 (Standardized Coef)')
    axes[1].set_xlabel('Coefficient Value')
    
    plt.tight_layout()
    # plt.show()

    return fig, results


def timeserise_pro(prof, df, wf_no):    
    
    df['WaferNo']=df['WaferNo'].str.split('_').str[3].astype('int64') # wafer no int 순으로 프린트 한다 
   
    if prof == 'setpoint/flow':
       fig=plt.figure(figsize=(10, 10)) 
       count=1
       for col  in df.columns[1:7]: 
           ax=fig.add_subplot(6, 2, count )
           count +=1           
           plt.plot(range(len(df)), df['flow_'+col]  , label='flow')
           plt.plot(range(len(df)), df[col]   ,  label='setpoint(recipe)')
           plt.title(col)
           

    elif 'flow/wafer' in prof  :
        fig=plt.figure(figsize=(10, 10)) 
        count=1
        for col  in df.columns[7:13]: 
            ax=fig.add_subplot(6, 2, count )
            count +=1  
            
            for w in wf_no.split(','):  
                w=int(w)
                w_df = df[df['WaferNo']==w][col]
                print(w,"==============")
                plt.plot(range(len(w_df)), w_df, label='run#'+str(w))
            
            plt.title(col)
            
            
        
        
        
        
    else  :
        fig=plt.figure(figsize=(10, 10)) 
        group_wafer = df.groupby('WaferNo')     #  print순서 1,11,12,13 .... 2,3,4, 수정 해야 할 듯함   
        i=1
        if prof=='wafer/setpoint':
            cols=df.columns[1:7]
        else:            
            cols=df.columns[7:13]
        for wid, df  in group_wafer: 
            df.columns[1:7]
            df.head(30)
            # 1. ax를 직접 생성
            ax = fig.add_subplot(6, 3, i )
            df.index = [x for x in range(len(df))]
            # 2. df.plot에 ax를 인자로 전달 (이게 핵심입니다!)
            
            df[cols].plot(kind='line', ax=ax, legend=False)  # ax를 사용 시에는 legend=False를 해야 한다
            
            i +=1
            
            ax.set_title(str(wid), fontsize=10)
            ax.set_xticks(range(0, 31, 5))
           
            # 마지막에 전체 Figure에 대한 범례를 하나만 생성합니다.
    
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='lower center',
               ncol=10,
               bbox_to_anchor=(0.5, 1.02), fontsize=12)

    plt.tight_layout()
    # 범례 공간 확보를 위해 오른쪽 여백 조정
    fig.subplots_adjust(right=0.9)
    
    return fig


'''
df_pecvd, df_waf, df_recipe=load_data()
df_recipe.columns

df_recipe['WaferNo']=df_recipe['WaferNo'].str.split('_').str[3].astype('int64')


# 
t_recipe = df_recipe.groupby('WaferNo')[df_recipe.columns[1:7]].mean()

t_recipe.to_csv('t_recipe.csv')


df_recipe.info()
fig=plt.figure(figsize=(12, 12)) # 3x6 배열이므로 가로를 길게 설정
i=1
for wid, df  in group_wafer: 
    print(df.head(30))
    break 

li1 = [       
 'chA_AO_mfc2_setpoint_Si2H6', # Si2H6
 'chA_AO_mfc3_setpoint_N2O',  # N2O O는 영문 대문자
 'chA_AO_mfc10_setpoint_TN2', #TN2
 'chA_VIR_APC_Setpoint',  # Pressure
 'USER RF SET',   #Power
 'chA_VIR_Heater_Temp_Set'  #Temp      
 ]  #Time


li2 = ['chA_AI_mfc2_flow_Si2H6', # Si2H6
       'chA_AI_mfc3_flow_N2O',   # N2O O는 영문 대문자
       'chA_AI_mfc10_flow_TN2', 
       'chA_AI_Manometer_Pressure',
       'FORWARD POWER',   #Power
       'chA_VIR_Heater_Temp'  #Temp  
       ]




timeserise_pro(li1, li2, df_pecvd)

'''    

def set_recipe(X, t_mean, t_uni,w_mean, w_uni ): 
    
        # 스케일링
    y_mean=X['mean']
    y_uni=X['uniformity']
    X=X.drop(['mean','uniformity'], axis=1)
    
    # ★ 타겟 변수에 log1p (log(1 + x)) 정규화 적용
    y_mean_log = np.log1p(y_mean)
    y_uni_log = np.log1p(y_uni)
   # 독립변수 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 데이터프레임 형태로 복원 (컬럼명 유지 및 상수항 추가를 위함)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # ★ statsmodels OLS 학습을 위해 반드시 상수항(const)을 추가해야 합니다.
    X_final_with_const = sm.add_constant(X_scaled_df)
    
    y_mean_log = np.array(y_mean_log).ravel()
    y_uni_log = np.array(y_uni_log).ravel()
    
    # ----------------------------------------------------
    # 2. 방법 A: 개별 sm.OLS 모델 학습
    # ----------------------------------------------------
    model_mean = sm.OLS(y_mean_log, X_final_with_const).fit()
    model_uni = sm.OLS(y_uni_log, X_final_with_const).fit()
    
    # (선택) 모델이 잘 학습되었는지 통계량 요약 확인
    # print(model_mean.summary())
    # print(model_uni.summary())
    
    # ----------------------------------------------------
    # 3. 목표치(Target) 및 목적 함수(Objective Function) 정의
    # ----------------------------------------------------
    # target_mean = 3300
    # target_uni = 3
    
    # 중요도 가중치 (Streamlit 슬라이더 값 반영 가능 / 여기서는 60%, 40% 예시)
    
    
    def objective_function(X_test_scaled):
        """
        최적화 알고리즘이 탐색할 X_test_scaled 1차원 배열을 받아 
        로그 변환된 OLS 모델로 예측한 후, 지수함수(expm1)로 역변환하여 
        원래 스케일에서의 오차(Loss)를 계산하는 함수
        """
        # 1. 최적화 알고리즘이 제시한 독립변수 배열 뒤에 상수항(1.0)을 붙여줌
        # statsmodels의 predict를 쓰려면 학습할 때와 동일하게 const(1)가 포함되어야 합니다.
        X_test_with_const = np.insert(X_test_scaled, 0, 1.0) 
        
        # 2. 각 모델의 래시피 예측값 계산
        pred_mean_log = model_mean.predict(X_test_with_const)[0]
        pred_uni_log = model_uni.predict(X_test_with_const)[0]
        # 3. ★ 중요: 원래 단위(3300, 3 등)로 복원하기 위해 expm1 역변환 적용
        pred_mean = np.expm1(pred_mean_log)
        pred_uni = np.expm1(pred_uni_log)
        # 3. 목표값과의 상대 오차 제곱 계산 (Scale 차이 보정)
        loss_mean = ((pred_mean - t_mean)/t_mean)  ** 2
        loss_uni = ((pred_uni - t_uni)/t_uni) ** 2
        
        # 4. 가중치가 반영된 최종 오차 합산
        total_loss = (w_mean/100 * loss_mean) + (w_uni/100 * loss_uni)
        return total_loss
    
    # ----------------------------------------------------
    # 4. 최적화 실행 (SLSQP 방식)
    # ----------------------------------------------------
    # 초기값 설정 (독립변수 개수만큼 0으로 채운 배열)
    initial_guess = np.zeros(X.shape[1])
    
    # 탐색 범위(Bounds) 제한: 실제 수집된 데이터의 최소/최대 범위를 벗어나지 않게 설정
    bounds = [(X_scaled[:, i].min(), X_scaled[:, i].max()) for i in range(X.shape[1])]
    
    # 최적화 시작
    res = minimize(objective_function, initial_guess, method='SLSQP', bounds=bounds)
    # res = minimize(objective_function, initial_guess)
    
    # ----------------------------------------------------
    # 5. 결과 해석 및 원래 공정 단위(역스케일링) 변환
    # ----------------------------------------------------
    # 최적화된 스케일링 상태의 X값 (1, N) 차원으로 변환
    best_X_scaled = res.x.reshape(1, -1)
    
    # 원래 물리적인 공정 단위 레시피로 복원
    best_X_original = scaler.inverse_transform(best_X_scaled)[0]
    
    # 해당 최적 레시피일 때 두 OLS 모델의 최종 예측 결과 확인
    best_X_with_const = np.insert(res.x, 0, 1.0)
    final_pred_mean = model_mean.predict(best_X_with_const)[0]
    final_pred_uni = model_uni.predict(best_X_with_const)[0]
    
    # 출력을 위한 데이터프레임 구성
    result_df = pd.DataFrame({
        'Feature': X.columns,
        'Optimal_Recipe_Value': best_X_original
    })
    
    print(f"🎯 목표치 설정 결과:")
    print(f"예측 Mean: {final_pred_mean:.2f} (Target: {t_mean}), (mean %:{w_mean})")
    print(f"예측 Uniformity: {final_pred_uni:.2f} (Target: {t_uni}), (uniformity % : {w_uni})\n")
    print("🛠️ 타겟 만족을 위한 최적의 독립변수 레시피 조합:")
    print(result_df)
        
    return result_df    
 

# %%

def wafer_pattern(df_waf, WAFER_ID):  # 리포트용 웨이퍼 ID
    
    # ════════════════════════════════════════════════════════
    # 0. 설정
    # ════════════════════════════════════════════════════════
          # 두께 컬럼명
    WAFER_RADIUS  = 150                     # 웨이퍼 반경 (mm), 300mm 웨이퍼=150
    SPEC_USL      = None                    # 두께 상한 스펙 (없으면 자동)
    SPEC_LSL      = None                    # 두께 하한 스펙 (없으면 자동)
    # ════════════════════════════════════════════════════════
    # 1. 데이터 로드
    # ════════════════════════════════════════════════════════
    # print("=" * 60)
    # print("1. 데이터 로드")
    # print("=" * 60)
    
    df = df_waf.copy()
    
    # print(f"  측정 포인트 수: {len(df)}")
    # print(df.head())
    
    x   = df['x'].values
    y   = df['y'].values
    thk = df[WAFER_ID].values
    
    
    # print(f"x   : {x.shape}")   # (N,) 이어야 함
    # print(f"y   : {y.shape}")   # (N,) 이어야 함
    # print(f"thk : {thk.shape}") # (N,) 이어야 함
    print(df.columns.tolist())
    
    
    # ════════════════════════════════════════════════════════
    # 2. 기본 통계
    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("2. 기본 통계 분석")
    print("=" * 60)
    
    mean_thk   = np.mean(thk)
    std_thk    = np.std(thk)
    min_thk    = np.min(thk)
    max_thk    = np.max(thk)
    range_thk  = max_thk - min_thk
    nu         = range_thk / (2 * mean_thk) * 100   # Non-Uniformity (%)
    skewness   = stats.skew(thk)
    kurt       = stats.kurtosis(thk)
    
    # 스펙 자동 계산
    if SPEC_USL is None:
        SPEC_USL = mean_thk + 3 * std_thk
    if SPEC_LSL is None:
        SPEC_LSL = mean_thk - 3 * std_thk
    
    # Cpk
    cpu = (SPEC_USL - mean_thk) / (3 * std_thk)
    cpl = (mean_thk - SPEC_LSL) / (3 * std_thk)
    cpk = min(cpu, cpl)
    cp  = (SPEC_USL - SPEC_LSL) / (6 * std_thk)
    
    # 불량 포인트
    defect_mask   = (thk < SPEC_LSL) | (thk > SPEC_USL)
    defect_count  = defect_mask.sum()
    defect_rate   = defect_count / len(thk) * 100
    
    print(f"  평균 두께  : {mean_thk:.1f} Å")
    print(f"  표준편차   : {std_thk:.1f} Å")
    print(f"  최소 두께  : {min_thk:.1f} Å")
    print(f"  최대 두께  : {max_thk:.1f} Å")
    print(f"  두께 범위  : {range_thk:.1f} Å")
    print(f"  불균일도(NU): {nu:.2f} %")
    print(f"  Cp         : {cp:.3f}")
    print(f"  Cpk        : {cpk:.3f}")
    print(f"  불량 포인트: {defect_count}개 ({defect_rate:.1f}%)")
    
    # ════════════════════════════════════════════════════════
    # 3. 패턴 분류 (개선 버전)
    # ════════════════════════════════════════════════════════
    
    # 반경 계산
    r = np.sqrt(x**2 + y**2)
    
    # 영역 분리
    center_mask = r < WAFER_RADIUS * 0.4
    edge_mask   = r > WAFER_RADIUS * 0.7
    mid_mask    = ~center_mask & ~edge_mask
    
    center_mean = np.mean(thk[center_mask]) if center_mask.sum() > 0 else mean_thk
    edge_mean   = np.mean(thk[edge_mask])   if edge_mask.sum()   > 0 else mean_thk
    mid_mean    = np.mean(thk[mid_mask])    if mid_mask.sum()    > 0 else mean_thk
    
    # 4분면 평균
    q_means = {
        '우상단(Q1)': np.mean(thk[(x > 0) & (y > 0)]) if ((x > 0) & (y > 0)).sum() > 0 else mean_thk,
        '좌상단(Q2)': np.mean(thk[(x < 0) & (y > 0)]) if ((x < 0) & (y > 0)).sum() > 0 else mean_thk,
        '좌하단(Q3)': np.mean(thk[(x < 0) & (y < 0)]) if ((x < 0) & (y < 0)).sum() > 0 else mean_thk,
        '우하단(Q4)': np.mean(thk[(x > 0) & (y < 0)]) if ((x > 0) & (y < 0)).sum() > 0 else mean_thk,
    }
    
    # ── 좌우 / 상하 / 대각선 성분 계산 ─────────────────────
    left_mean   = np.mean(thk[x < 0]) if (x < 0).sum() > 0 else mean_thk
    right_mean  = np.mean(thk[x > 0]) if (x > 0).sum() > 0 else mean_thk
    top_mean    = np.mean(thk[y > 0]) if (y > 0).sum() > 0 else mean_thk
    bottom_mean = np.mean(thk[y < 0]) if (y < 0).sum() > 0 else mean_thk
    
    lr_diff = abs(left_mean  - right_mean)
    tb_diff = abs(top_mean   - bottom_mean)
    
    # 대각선 = 대각 차이에서 좌우/상하 성분을 제거한 순수 대각 성분
    diag1      = abs(q_means['우상단(Q1)'] - q_means['좌하단(Q3)'])
    diag2      = abs(q_means['좌상단(Q2)'] - q_means['우하단(Q4)'])
    diag_raw   = max(diag1, diag2)
    diag_score = max(diag_raw - max(lr_diff, tb_diff), 0)  # 순수 대각선 성분
    
    # ── t-test 유의성 검정 ───────────────────────────────────
    from scipy.stats import ttest_ind
    
    _, p_lr = ttest_ind(thk[x < 0], thk[x > 0])
    _, p_tb = ttest_ind(thk[y > 0], thk[y < 0])
    
    # 유의하지 않으면 점수를 0으로
    if p_lr >= 0.05:
        lr_diff = 0
    if p_tb >= 0.05:
        tb_diff = 0
    
    # ── 패턴 점수 등록 ───────────────────────────────────────
    pattern_scores = {}
    
    # 1) Center 패턴
    pattern_scores['Center (중앙 두꺼움)']  = max(center_mean - edge_mean, 0)
    
    # 2) Edge 패턴
    pattern_scores['Edge (엣지 두꺼움)']    = max(edge_mean - center_mean, 0)
    
    # 3) Donut 패턴
    pattern_scores['Donut (도넛형)']        = max(mid_mean - max(center_mean, edge_mean), 0)
    
    # 4) 좌우 비대칭 (통계 검정 후)
    pattern_scores['Left-Right (좌우 비대칭)'] = lr_diff
    
    # 5) 상하 비대칭 (통계 검정 후)
    pattern_scores['Top-Bottom (상하 비대칭)'] = tb_diff
    
    # 6) 순수 대각선 편향
    pattern_scores['Diagonal (대각선 편향)']   = diag_score
    
    # ── 패턴 판정 ────────────────────────────────────────────
    main_pattern     = max(pattern_scores, key=pattern_scores.get)
    pattern_strength = pattern_scores[main_pattern]
    
    # ── 방향 정보 추가 ───────────────────────────────────────
    direction = ""
    if main_pattern == 'Left-Right (좌우 비대칭)':
        direction = '좌측 두꺼움' if left_mean > right_mean else '우측 두꺼움'
    elif main_pattern == 'Top-Bottom (상하 비대칭)':
        direction = '상단 두꺼움' if top_mean > bottom_mean else '하단 두꺼움'
    elif main_pattern == 'Center (중앙 두꺼움)':
        direction = f'중앙({center_mean:.1f}Å) vs 엣지({edge_mean:.1f}Å)'
    elif main_pattern == 'Edge (엣지 두꺼움)':
        direction = f'엣지({edge_mean:.1f}Å) vs 중앙({center_mean:.1f}Å)'
    elif main_pattern == 'Diagonal (대각선 편향)':
        thick_q = max(q_means, key=q_means.get)
        thin_q  = min(q_means, key=q_means.get)
        direction = f'{thick_q} 두꺼움 / {thin_q} 얇음'
    
    # ── 결과 출력 ────────────────────────────────────────────
    print(f"\n  [영역별 평균]")
    print(f"    중앙  : {center_mean:.1f} Å  ({center_mask.sum()}포인트)")
    print(f"    중간  : {mid_mean:.1f} Å  ({mid_mask.sum()}포인트)")
    print(f"    엣지  : {edge_mean:.1f} Å  ({edge_mask.sum()}포인트)")
    print(f"\n  [비대칭 분석]")
    print(f"    좌({left_mean:.1f}) vs 우({right_mean:.1f}) → 차이={lr_diff:.1f}Å  p={p_lr:.4f}")
    print(f"    상({top_mean:.1f}) vs 하({bottom_mean:.1f}) → 차이={tb_diff:.1f}Å  p={p_tb:.4f}")
    print(f"    대각선 순수 성분 : {diag_score:.1f} Å")
    print(f"\n  [패턴 점수 전체]")
    for k, v in sorted(pattern_scores.items(), key=lambda x: x[1], reverse=True):
        bar = '█' * int(v / 10)
        print(f"    {k:30s}: {v:6.1f} Å  {bar}")
    print(f"\n  ★ 최종 패턴  : {main_pattern}")
    print(f"  ★ 방향/특징  : {direction}")
    print(f"  ★ 패턴 강도  : {pattern_strength:.1f} Å")
    
    # ════════════════════════════════════════════════════════
    # 4. 원인 추정
    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("4. 불량 원인 추정")
    print("=" * 60)
    
    CAUSE_MAP = {
        'Center (중앙 두꺼움)': {
            '원인 1': 'RF Power 과다 (중앙 플라즈마 집중)',
            '원인 2': '샤워헤드-웨이퍼 간격 좁음',
            '원인 3': 'Chuck 온도 불균일 (중앙 고온)',
            '조치':   'RF Power 감소 / 샤워헤드 Gap 조정',
        },
        'Edge (엣지 두꺼움)': {
            '원인 1': '엣지 링(Focus Ring) 마모',
            '원인 2': '가스 흐름이 외곽으로 편중',
            '원인 3': 'Chuck 온도 불균일 (엣지 고온)',
            '조치':   'Focus Ring 교체 / 가스 유량 재조정',
        },
        'Donut (도넛형)': {
            '원인 1': '샤워헤드 중앙 홀 막힘',
            '원인 2': 'RF 전극 중앙 손상',
            '원인 3': 'Precursor 분포 불균일',
            '조치':   '샤워헤드 세정/교체 / RF 전극 점검',
        },
        'Diagonal (대각선 편향)': {
            '원인 1': 'Chuck Tilt (기울어짐)',
            '원인 2': '가스 공급 방향 편심',
            '원인 3': 'RF 전극 편심 장착',
            '조치':   'Chuck 레벨링 점검 / 가스라인 대칭성 확인',
        },
        'Left-Right (좌우 비대칭)': {
            '원인 1': '가스 공급 포트 위치 편심 (좌우)',
            '원인 2': 'Chamber 내부 비대칭 구조',
            '원인 3': 'Chuck 좌우 온도 차이',
            '조치':   '가스 공급 포트 점검 / Chuck 온도 프로파일 확인',
        },
        'Top-Bottom (상하 비대칭)': {
            '원인 1': '가스 공급 포트 위치 편심 (상하)',
            '원인 2': 'Pump 위치에 의한 가스 흐름 편향',
            '원인 3': 'Chuck 상하 온도 차이',
            '조치':   'Pump 밸런스 점검 / 가스 흐름 시뮬레이션',
        },
    }
    
    causes = CAUSE_MAP.get(main_pattern, {})
    for k, v in causes.items():
        print(f"  {k}: {v}")
    
    # Cpk 판정
    if cpk >= 1.67:
        cpk_judge = "매우 우수 ✅"
    elif cpk >= 1.33:
        cpk_judge = "양호 ✅"
    elif cpk >= 1.00:
        cpk_judge = "보통 ⚠️ (개선 권고)"
    else:
        cpk_judge = "불량 🔴 (즉각 조치)"
    
    print(f"\n {WAFER_ID} :  Cpk={cpk:.3f} → {cpk_judge}")
    
    # ════════════════════════════════════════════════════════
    # 5. 시각화
    # ════════════════════════════════════════════════════════
    print("\n시각화 저장 중...")
    
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor('#0d1117')
    #fig.patch.set_facecolor('white')
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.4)
    
    CMAP = 'RdYlBu_r'
    TEXT_COLOR = 'white'
    GRID_COLOR = '#30363d'
    
    # 공통 스타일
    def style_ax(ax, title):
        ax.set_facecolor('#161b22')
        ax.set_title(title, color=TEXT_COLOR, fontweight='bold', fontsize=10, pad=8)
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)
        ax.grid(color=GRID_COLOR, linewidth=0.5, alpha=0.5)
    
    # 그리드 보간
    xi = np.linspace(-WAFER_RADIUS, WAFER_RADIUS, 200)
    yi = np.linspace(-WAFER_RADIUS, WAFER_RADIUS, 200)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), thk, (Xi, Yi), method='cubic')
    
    # 웨이퍼 원 마스크
    mask_circle = (Xi**2 + Yi**2) > WAFER_RADIUS**2
    Zi[mask_circle] = np.nan
    
    # (1) 컨투어 맵
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    style_ax(ax1, f'웨이퍼 두께 컨투어 맵\n{WAFER_ID}')
    cf = ax1.contourf(Xi, Yi, Zi, levels=20, cmap=CMAP, alpha=0.9)
    cs = ax1.contour(Xi, Yi, Zi, levels=10, colors='white', alpha=0.3, linewidths=0.5)
    ax1.clabel(cs, inline=True, fontsize=6, colors='white', fmt='%.0f')
    cb = plt.colorbar(cf, ax=ax1)
    cb.set_label('두께 (Å)', color=TEXT_COLOR, fontsize=9)
    cb.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)
    ax1.scatter(x[~defect_mask], y[~defect_mask], c='white', s=20, alpha=0.6, zorder=5, label='정상')
    ax1.scatter(x[defect_mask],  y[defect_mask],  c='red',   s=40, alpha=0.9, zorder=6,
                marker='x', linewidths=2, label=f'불량({defect_count}개)')
    circle = Circle((0,0), WAFER_RADIUS, fill=False, color='cyan', linewidth=1.5, linestyle='--')
    ax1.add_patch(circle)
    ax1.set_xlim(-WAFER_RADIUS*1.1, WAFER_RADIUS*1.1)
    ax1.set_ylim(-WAFER_RADIUS*1.1, WAFER_RADIUS*1.1)
    ax1.set_aspect('equal')
    ax1.set_xlabel('X Position (mm)', color=TEXT_COLOR, fontsize=9)
    ax1.set_ylabel('Y Position (mm)', color=TEXT_COLOR, fontsize=9)
    ax1.legend(fontsize=8, facecolor='#21262d', labelcolor=TEXT_COLOR, loc='upper right')
    
    # (2) 두께 분포 히스토그램
    ax2 = fig.add_subplot(gs[0, 2])
    style_ax(ax2, '두께 분포')
    n, bins, patches = ax2.hist(thk, bins=20, color='#58a6ff', edgecolor='#0d1117', alpha=0.8)
    ax2.axvline(mean_thk,  color='yellow', linewidth=2, linestyle='-',  label=f'평균 {mean_thk:.0f}')
    ax2.axvline(SPEC_USL,  color='red',    linewidth=1.5, linestyle='--', label=f'USL {SPEC_USL:.0f}')
    ax2.axvline(SPEC_LSL,  color='red',    linewidth=1.5, linestyle='--', label=f'LSL {SPEC_LSL:.0f}')
    ax2.set_xlabel('두께 (Å)', color=TEXT_COLOR, fontsize=8)
    ax2.legend(fontsize=7, facecolor='#21262d', labelcolor=TEXT_COLOR)
    
    # (3) 반경별 두께 산점도
    ax3 = fig.add_subplot(gs[0, 3])
    style_ax(ax3, '반경 vs 두께')
    # sc = ax3.scatter(r, thk, c=thk, cmap=CMAP, s=30, alpha=0.7)
    ax3.axhline(mean_thk, color='yellow', linewidth=1.5, linestyle='--', label='평균')
    ax3.axhline(SPEC_USL, color='red',    linewidth=1,   linestyle=':',  label='USL/LSL')
    ax3.axhline(SPEC_LSL, color='red',    linewidth=1,   linestyle=':')
    z = np.polyfit(r, thk, 2)
    p = np.poly1d(z)
    r_line = np.linspace(0, WAFER_RADIUS, 100)
    ax3.plot(r_line, p(r_line), 'cyan', linewidth=2, label='추세선')
    ax3.set_xlabel('반경 (mm)', color=TEXT_COLOR, fontsize=8)
    ax3.set_ylabel('두께 (Å)', color=TEXT_COLOR, fontsize=8)
    ax3.legend(fontsize=7, facecolor='#21262d', labelcolor=TEXT_COLOR)
    
    # (4) 4분면 평균 두께
    ax4 = fig.add_subplot(gs[1, 2])
    style_ax(ax4, '4분면 평균 두께')
    q_labels = list(q_means.keys())
    q_values = list(q_means.values())
    colors_q = ['#e74c3c' if v == max(q_values) else
                '#2ecc71' if v == min(q_values) else '#58a6ff' for v in q_values]
    bars = ax4.bar(q_labels, q_values, color=colors_q, edgecolor='#0d1117')
    ax4.axhline(mean_thk, color='yellow', linewidth=1.5, linestyle='--', label='전체 평균')
    ax4.set_ylim(min(q_values)*0.998, max(q_values)*1.002)
    ax4.tick_params(axis='x', rotation=20, labelsize=7)
    for bar, val in zip(bars, q_values):
        ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                 f'{val:.0f}', ha='center', va='bottom', color=TEXT_COLOR, fontsize=7)
    ax4.legend(fontsize=7, facecolor='#21262d', labelcolor=TEXT_COLOR)
    
    # (5) Z-score 맵 (이상치 위치)
    ax5 = fig.add_subplot(gs[1, 3])
    style_ax(ax5, 'Z-Score 맵 (이상치 위치)')
    z_scores = (thk - mean_thk) / std_thk
    Zi_z = griddata((x, y), z_scores, (Xi, Yi), method='cubic')
    Zi_z[mask_circle] = np.nan
    cf5 = ax5.contourf(Xi, Yi, Zi_z, levels=np.linspace(-3, 3, 20),
                       cmap='RdBu_r', alpha=0.9)
    plt.colorbar(cf5, ax=ax5).ax.yaxis.set_tick_params(color=TEXT_COLOR)
    circle5 = Circle((0,0), WAFER_RADIUS, fill=False, color='cyan', linewidth=1.5, linestyle='--')
    ax5.add_patch(circle5)
    ax5.set_aspect('equal')
    ax5.set_xlim(-WAFER_RADIUS*1.1, WAFER_RADIUS*1.1)
    ax5.set_ylim(-WAFER_RADIUS*1.1, WAFER_RADIUS*1.1)
    ax5.set_xlabel('X (mm)', color=TEXT_COLOR, fontsize=8)
    
    # (6) 자동 리포트 텍스트
    # ax6 = fig.add_subplot(gs[2, :])
    # ax6.set_facecolor('#161b22')
    # # ax6.axis('off')
    # for spine in ax6.spines.values():
    #     spine.set_edgecolor(GRID_COLOR)
    
    report_lines = [
        f"{'='*90}",
        f"  PECVD 웨이퍼 두께 불량 자동 분석 리포트  │  {WAFER_ID}",
        f"{'='*90}",
        f"  [기본 통계]  평균={mean_thk:.1f}Å  │  Std={std_thk:.1f}Å  │  Min={min_thk:.1f}Å  │  Max={max_thk:.1f}Å  │  범위={range_thk:.1f}Å  │  NU={nu:.2f}%",
        f"  [공정 능력]  Cp={cp:.3f}  │  Cpk={cpk:.3f}  →  {cpk_judge}",
        f"  [불량 현황]  LSL={SPEC_LSL:.1f}Å  │  USL={SPEC_USL:.1f}Å  │  불량 포인트={defect_count}개 ({defect_rate:.1f}%)",
        f"  {'─'*88}",
        f"  [패턴 진단]  ★ {main_pattern}  (강도: {pattern_strength:.1f}Å)",
        f"  [원인 1]    {causes.get('원인 1', '-')}",
        f"  [원인 2]    {causes.get('원인 2', '-')}",
        f"  [원인 3]    {causes.get('원인 3', '-')}",
        f"  [권고 조치]  {causes.get('조치', '-')}",
    ]
    
    report_text = "\n".join(report_lines)
    # ax6.text(0.01, 0.95, report_text, transform=ax6.transAxes,
    #          fontsize=8.5, verticalalignment='top', color=TEXT_COLOR,
    #          fontfamily='Malgun Gothic',
    #          bbox=dict(boxstyle='round', facecolor='#21262d', alpha=0.8))
    
    fig.suptitle('PECVD SiN 웨이퍼 두께 불량 패턴 분석', color=TEXT_COLOR,
                 fontsize=15, fontweight='bold', y=0.98)
    
    # plt.savefig("output/wafer_defect_report.png", dpi=150, bbox_inches='tight',facecolor='#0d1117')
    # print("  → wafer_defect_report.png 저장 완료")
    plt.title(WAFER_ID)
    # plt.show()
    
    # ════════════════════════════════════════════════════════
    # 6. TXT 리포트 저장
    # ════════════════════════════════════════════════════════
    viewtxt=''
    with open("output/wafer_defect_report.txt", "w", encoding="utf-8-sig") as f:
        f.write("=" * 60 + "\n")
        f.write(f"PECVD 웨이퍼 두께 불량 자동 분석 리포트\n")
        f.write(f"Wafer ID : {WAFER_ID}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"[기본 통계]\n")
        f.write(f"  평균 두께   : {mean_thk:.1f} Å\n")
        f.write(f"  표준편차    : {std_thk:.1f} Å\n")
        f.write(f"  최소 두께   : {min_thk:.1f} Å\n")
        f.write(f"  최대 두께   : {max_thk:.1f} Å\n")
        f.write(f"  두께 범위   : {range_thk:.1f} Å\n")
        f.write(f"  불균일도(NU): {nu:.2f} %\n")
        f.write(f"  Skewness    : {skewness:.3f}\n")
        f.write(f"  Kurtosis    : {kurt:.3f}\n\n")
        f.write(f"[공정 능력]\n")
        f.write(f"  LSL : {SPEC_LSL:.1f} Å\n")
        f.write(f"  USL : {SPEC_USL:.1f} Å\n")
        f.write(f"  Cp  : {cp:.3f}\n")
        f.write(f"  Cpk : {cpk:.3f}  →  {cpk_judge}\n\n")
        f.write(f"[불량 현황]\n")
        f.write(f"  불량 포인트 : {defect_count}개 / {len(thk)}개 ({defect_rate:.1f}%)\n\n")
        f.write(f"[4분면 분석]\n")
        for k, v in q_means.items():
            f.write(f"  {k}: {v:.1f} Å\n")
           
        f.write(f"\n[패턴 진단]\n")
        
        f.write(f"  ★ 주요 패턴 : {main_pattern}\n")
       
        f.write(f"  ★ 패턴 강도 : {pattern_strength:.1f} Å\n\n")
        
        
        f.write(f"[원인 추정]\n")
        for k, v in causes.items():
            f.write(f"  {k}: {v}\n")
            
        
    
    # print("  → wafer_defect_report.txt 저장 완료")
    return fig, report_lines

        
# %%





def wafer_multi_model(df, len_col):    
        
        tx = df.iloc[:, :len_col]        
        ty = df.index
        # 정규화 X
        scaler   = StandardScaler()
        X = scaler.fit_transform(tx)       
        
        # 정규화 y
        encoder = OneHotEncoder(sparse_output=False)        # 3. 학습 및 변환
        ty= np.array(ty)
        ty=ty.reshape(-1, 1)
        y = encoder.fit_transform(ty)    
       
        
        TX_train, X_test, Ty_train, y_test \
        = train_test_split(X, y, test_size=0.3, random_state=42)
        TX_train, X_val, Ty_train, y_val \
        = train_test_split(TX_train, Ty_train, test_size=0.2, random_state=42)

       
        
        # 2. 데이터 증강 함수 (Gaussian Noise 추가)
        def augment_data(X, y, multiplier=50):
            X_augmented = [X]
            y_augmented = [y]
            
            for _ in range(multiplier):
                # 원본 데이터에 미세한 노이즈(표준편차 0.05) 추가
                noise = np.random.normal(0, 0.05, X.shape)
                X_augmented.append(X + noise)
                y_augmented.append(y)
                
            return np.vstack(X_augmented), np.concatenate(y_augmented)
        
        # 18개 데이터를 918개(18 + 18*50)로 증강
        # 과적합이 예상 된다 :  데이터가 없어서 전체를 증강 하여야 한다 
        X_train, y_train = augment_data(X, y, multiplier=50)
        
        # 3. 모델 구성
        model = models.Sequential([
            layers.Input(shape=(len_col,)),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(), # 데이터가 적을 때 안정화에 도움
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dense(18, activation='softmax') # 최종 클래스 18개
        ])
        
        # 4. 컴파일 및 학습
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 증강된 데이터로 학습
        model.fit(X_train, y_train, epochs=100, batch_size=16, shuffle=True
                  , validation_data=(X_val, y_val)) # 직접 검증 데이터 지정)
        
        # 5. 테스트 (1개 행 예측)
        # test_row = X_test[0:1] # 첫 번째 행을 테스트용으로 사용
        prediction = model.predict(X_test)
        t_test=np.argmax(y_test, axis=1)
        t_pred=np.argmax(prediction, axis=1)
        
        # print(t_test)
        # print(f"예측 결과 클래스: {t_pred}")
        
        li=[]
        li.append(f"label    : {t_test}")
        li.append(f"예측 결과 : {t_pred}")
        
         
                
        # 4. 결과 시각화 (Subplot 활용)
        fig = plt.figure(figsize=(20, 10))
        
        # [차트 1] Accuracy 비교 막대 그래프
        plt.subplot(1, 2, 1)      
        plt.scatter(t_test, t_pred, color='b')# 기준선 (y=x) 그리기
        line_coords = [t_test.min(), t_test.max()]
        plt.plot(line_coords, line_coords, color='red', linestyle='--', lw=2, label='Perfect Prediction')
        plt.title("Multi-class Accuracy Comparison", fontsize=15)
     
        
        # [차트 2] 성능이 가장 좋은 모델의 Confusion Matrix
       
        plt.subplot(1,2,2)
        cm = confusion_matrix(t_test, t_pred) 
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=t_test, 
                    yticklabels=t_pred)
        plt.title('Confusion Matrix:')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        # plt.show()
        return fig, li
        # # 상세 리포트 출력
        # print(f"--- Best Model ({best_model_name}) Detail Report ---")
       
# t_waf=df_waf.iloc[:,3:-1].T
# df_wafer_setpoint
    
#wafer_multi_model(df_wafer_setpoint, 7)