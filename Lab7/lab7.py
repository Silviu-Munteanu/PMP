import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az
import math
from scipy import stats
if __name__ == '__main__':
     data=pd.read_csv("Prices.csv")
     Price=data['Price'].values
     Speed=data['Speed'].values
     HardDrive = data['HardDrive'].values
     Ram=data['Ram'].values
     Premium=data['Premium'].values
     model=pm.Model()
     #ex1
     with pm.Model() as model:
          alpha=pm.Normal('alpha',mu=0,sd=1)
          beta1=pm.Normal('beta1',mu=0,sd=1)
          beta2=pm.Normal('beta2',mu=0,sd=1)
          sigma=pm.HalfNormal('sigma',sd=1)
          mu = pm.Deterministic('mu', alpha + beta1 * Speed + beta2 * [math.log(i) for i in HardDrive])
          Price_obs=pm.Normal('Price_obs', mu=mu,sd=sigma,observed=Price)
          trace = pm.sample(10, tune=10,return_inferencedata=True)
     az.plot_posterior(trace)
     #ex2
     #az.plot_posterior({"beta1": trace['beta1'], "beta2": trace['beta2']},hdi_prob=0.95)
     plt.show()
