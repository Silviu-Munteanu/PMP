import numpy
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import math
import random
import arviz as az
time=[]
time1= stats.expon.rvs(0,1/4,size=10000)
time2= stats.expon.rvs(0,1/6,size=10000)
print(len(time1))
for i in range(10000):
    if random.random()<0.4:
        time.append(time1[i])
    else:
        time.append(time2[i])
az.plot_posterior({'M1':time1,'M2':time2,'AVG':time})
plt.show()