import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az
import math
from scipy import stats
if __name__ == '__main__':
     data=pd.read_csv("Admission.csv")
     admission=data['Admission'].values
     gre=data['GRE'].values
     gpa = data['GPA'].values
     model=pm.Model()
     with model:
          alpha = pm.Normal('alpha', mu=0, sd=1)
          beta1 = pm.Normal('beta1', mu=0, sd=1)
          beta2 = pm.Normal('beta2', mu=0, sd=1)
          sigma = pm.HalfNormal('sigma', sd=1)
          mu=pm.Deterministic('mu', pm.math.sigmoid(alpha + beta1 * gre + beta2 * gpa))
          bd1 = pm.Deterministic('bd1', (alpha / beta1) * -1)
          bd2 = pm.Deterministic('bd2', (alpha / beta2) * -1)
          admission_obs = pm.Normal('admission_obs', mu=mu, sd=sigma, observed=admission)
          trace = pm.sample(50, tune=50, return_inferencedata=True)
     posterior_0 = trace.posterior.stack(samples=("chain", "draw"))
     theta = posterior_0['mu'].mean("samples")
     idx = np.argsort(gre)
     plt.plot(gre[idx], theta[idx], color='C2', lw=3)
     plt.vlines(posterior_0['bd1'].mean(), 0, 1, color='k')
     bd1_hpd = az.hdi(posterior_0['bd1'].values)
     plt.fill_betweenx([0, 1], bd1_hpd[0], bd1_hpd[1], color='k', alpha=0.5)
     plt.scatter(gre, np.random.normal(admission, 0.02),
                 marker='.', color=[f'C{x}' for x in admission])
     az.plot_hdi(gre, posterior_0['mu'].T, color='C2', smooth=False)
     plt.xlabel("gre")
     plt.ylabel('mu', rotation=0)
     locs, _ = plt.xticks()
     plt.xticks(locs, np.round(locs + gre.mean(), 1))
     plt.show()
