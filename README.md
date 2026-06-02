
plt.figure(figsize=(12, 6))
fig, ax1 = plt.subplots()
col1='총발전량'
col2='수력'

ax1.bar(df.index, df[col1], label=col1)
ax2=ax1.twinx()
ax2.plot(df.index, df[col2], color='r', marker='o',
         label=col2)  ###  2

ax1.set_xticks(df.index)
ax1.set_xticklabels(df.index, rotation=90)  # <- 여기서 회전 설정!
#   y축의 좌표값의 color를 blue로 수정 하는 방법
ax1.tick_params(axis='y', labelcolor='b') ### 1
ax1.set_ylabel(col1, color='b')
#   y축의 좌표값의 color를 red로 수정 하는 방법
ax2.tick_params(axis='y', labelcolor='r')
ax2.set_ylabel(col2, color='r')

ax1.set_ylim(0, df[col1].max()*1.2)  # ax1  y좌표의 구간을 표시한다 
ax2.set_ylim(df[col2].min()*0.5, df[col2].max()*1.2) # ax2 y좌표의 구간을 표시한다 
plt.title("북한 전력 발전량(1990 ~ 2016)")
plt.tight_layout()
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
plt.legend(lines1+lines2,
           labels1+labels2)
plt.show()




# 과제 업로드

260601 : https://drive.google.com/drive/folders/1xeNgq85aDo17UzSHX6fi-YslFNb1rszB?usp=sharing
