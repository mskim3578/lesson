
#   여러개 그래프
import statistics
plt.figure(figsize=(12,8))
chart_count=10
col=3
row=chart_count//col+1
count=1
for i in range(chart_count):
    x=np.linspace(0, 10, 500)
    y=np.random.rand(500)
    tmean= sum(y) / len(y)
    tstd = statistics.stdev(y)
    
    plt.subplot(row, col, count)
    plt.axhline(y=tmean, color='blue')
    plt.axhline(y=(tmean+(tstd*3)), color='r')
    plt.axhline(y=(tmean-(tstd*3)), color='r')
    count +=1
    plt.plot(x,y, color='blue', label=i)
    plt.title(f'EDA 데이터 탐색 - {i}')
    plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

