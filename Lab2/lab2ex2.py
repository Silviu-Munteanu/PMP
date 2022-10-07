import numpy
from scipy import stats
import matplotlib.pyplot as plt
import math
import random
import arviz as az
time=[]
proc_time1= stats.gamma.rvs(4,0,1/3,size=10000)
proc_time2= stats.gamma.rvs(4,0,1/2,size=10000)
proc_time3= stats.gamma.rvs(5,0,1/2,size=10000)
proc_time4= stats.gamma.rvs(5,0,1/3,size=10000)
lat=stats.expon.rvs(0,1/4,size=10000)
for i in range(10000):
    if random.random()<0.25:
        time.append(proc_time1[i]+lat[i])
    elif random.random()<0.5:
        time.append(proc_time2[i]+lat[i])
    elif random.random()<0.8:
        time.append(proc_time3[i] + lat[i])
    else:
        time.append(proc_time4[i]+lat[i])
az.plot_posterior({'latency':lat,'server1:':proc_time1,'server2':proc_time2,'server3':proc_time3,'server4':proc_time4,'AVG total':time})
plt.show()