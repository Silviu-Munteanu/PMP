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
       statie=pm.Exponential('S',1/8) #mean= 8 min
       trace=pm.sample(sample_s)
    az.plot_posterior(trace)
    left=0
    right=15
    p=0
    alpha=0
    r=0
    while left<right-0.1:
        alpha=(left+right)/2
        statie=stats.expon.rvs(alpha-1,size=sample_s) # stats.expon(alpha) are mean alpha +1
        less=0
        for i in range(1000):
            if statie[i] + trace['P'][i] < 15:
                less+=1
        p=less/sample_s
        if p < 0.95:
            right = (alpha + right) / 2
        else:
            left = (alpha + left) / 2
    mean=0
    print(alpha)
    stats.expon.mean(alpha-1) + stats.norm.mean(1,0.5)   #mean fara sampleuri
    ex=0
    norm=0
    ex+= stats.expon.rvs(alpha-1,size=sample_s)
    norm+= stats.norm.rvs(1,0.5,size=sample_s)
    mean = stats.expon.mean(alpha - 1) + stats.norm.mean(1, 0.5)
    print(mean)
    for i in range(1000):
        mean+=ex[i] + norm[i]
    mean/=1000
    print(mean)  #mean cu sampleuri
    plt.show()


