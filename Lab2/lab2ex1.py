import numpy
from scipy import stats
import matplotlib.pyplot as plt
import math
import random
import arviz as az
time1= stats.expon.rvs(0,1/4,size=10000)
time2= stats.expon.rvs(0,1/6,size=10000)
time=time1 * 0.4 + time2 * 0.6
az.plot_posterior({'M1':time1,'M2':time2,'AVG':time})
plt.show()