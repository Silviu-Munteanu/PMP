import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az
import scipy.stats as sc
if __name__ == '__main__':
    #A
    fig, axs = plt.subplots(2, 2)
    cafenea = sc.poisson.rvs(10, size=1000)
    axs[0, 0].hist(cafenea)  # pare mai folositoare o histograma
    caini = sc.uniform.rvs(5, 45, size=1000)
    axs[0, 1].hist(caini)  # rase de caini mai mici si mai mari
    elefanti = sc.norm.rvs(5000, 750, size=1000)
    axs[1, 0].hist(elefanti)
    oameni = sc.skewnorm.rvs(1, 75, 20, size=1000)
    axs[1, 1,].hist(oameni)
    model1 = pm.Model()
    theta = 0.2
    plt.show()
    #B
    fix , axs = plt.subplots(2,3)
    with model1:
        n = pm.Poisson('N', 10)
        y = pm.Binomial('Y', n, theta)
        trace = pm.sample(1000)
    for index,j in enumerate([0, 5, 10]):
        trace1=[t['N'] for t in trace if t['Y'] == j]
        axs[0][index].plot(trace1)
    model2 = pm.Model()
    theta=0.5
    with model2:
        n = pm.Poisson('N2', 10)
        y = pm.Binomial('Y2', n, theta)
        trace = pm.sample(1000)
    for index,j in enumerate([0, 5, 10]):
        trace2=[t['N2'] for t in trace if t['Y2'] == j]
        axs[1][index].plot(trace2)
    #pt theta mic si y mare N ul o sa tinda sa fie mare, ca sa compenseze pentru sansele mici de a fi generati y clienti din n cu probabilitate mica
    plt.show()
    data=np.random.binomial(n=1,p=0.5,size=150)
    #C
    model=pm.Model()
    with model:
        p = pm.Beta('p', 1., 1.)
        w = pm.Binomial('w', n=1, p=p, observed=data)
        trace = pm.sample(1000)
    model_m = pm.Model()
    with model_m:
        p = pm.Beta('p', 1., 1.)
        w = pm.Binomial('w', n=1, p=p, observed=data)
        trace_m = pm.sample(1000,step=pm.Metropolis())
    print(az.rhat(trace),az.rhat(trace_m))
    print(az.ess(trace), az.ess(trace_m))
    az.plot_autocorr(trace)
    az.plot_autocorr(trace_m)
    az.plot_trace(trace)
    az.plot_trace(trace_m)
    #rezultatele par mai bune la cel default, algoritmul default ar trebui sa fie NUTS
    plt.show()
