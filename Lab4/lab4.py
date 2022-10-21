import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az
from scipy import stats
if __name__ == '__main__':
    sample_s=1000
    model = pm.Model()
    with model:
       clienti=pm.Poisson('C',20)
       plasare=pm.Normal('P',1,0.5)
       statie=pm.Exponential('S',1/8) #alpha= 8 min
       trace=pm.sample(sample_s)
    az.plot_posterior(trace)
    left=0
    right=15
    p=0
    alpha=0
    r=0
    while left<right-0.1:
        alpha=(left+right)/2
        statie=stats.expon.rvs(1/alpha,size=sample_s)
        less=0
        for i in range(1000):
            if statie[i] + trace['P'][i] < 15:
                less+=1
        p=less/sample_s
        if p < 0.95:
            left=(alpha+right)/2
        else:
            right=(alpha+left)/2
        print(alpha)
    plt.show()


