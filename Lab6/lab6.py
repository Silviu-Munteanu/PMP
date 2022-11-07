import pymc3 as pm
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import arviz as az
import math
if __name__ == '__main__':
     f=open("data.csv","r")
     s=f.readlines()
     #print(s)
     ppvt=[]
     educ_cat=[]
     momage=[]
     for i in s[1:]:
          ppvt.append(int(i.split(",")[1]))
          educ_cat.append(int(i.split(",")[2]))
          momage.append(int(i.split(",")[3]))
     az.plot_posterior({'ppvt':ppvt,'educ_cat':educ_cat, "momage": momage})
     sum=0
     for i in ppvt:
          sum+=i
     avg_ppvt=sum/400
     sum=0
     for i in educ_cat:
          sum+=i
     avg_educ_cat=sum/400
     sum=0
     for i in momage:
          sum+=i
     avg_momage=sum/400
     sum=0
     for i in ppvt:
          sum+=(i-avg_ppvt) ** 2
     stdev_ppvt=math.sqrt(sum)
     sum=0
     for i in educ_cat:
          sum+=(i-avg_educ_cat) ** 2
     stdev_educ_cat=math.sqrt(sum)
     sum=0
     for i in momage:
          sum+=(i-avg_momage) ** 2
     stdev_momage=math.sqrt(sum)
     #ppvt=[(i-avg_ppvt) / stdev_ppvt for i in ppvt]
     #momage=[(i-avg_momage) / stdev_momage for i in momage]
     #educ_cat=[(i-avg_momage) / stdev_educ_cat for i in educ_cat]
     model=pm.Model()
     with model:
          alpha = pm.Normal("alpha", mu=avg_ppvt, sigma=stdev_ppvt)
          beta = pm.Normal("beta", mu=avg_momage, sigma=stdev_momage)
          sigma = pm.HalfNormal("sigma", sigma=stdev_ppvt)
          # Expected value of outcome
          mu = pm.Deterministic('μ', alpha + beta * momage)
          # Likelihood (sampling distribution) of observations
          Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=ppvt)
          idata_g = pm.sample(1000, tune=1000, return_inferencedata=True)
     map_estimate = pm.find_MAP(model=model)
     print(map_estimate)
     ppc = pm.sample_posterior_predictive(idata_g, samples=100, model=model)
     plt.plot(momage, alpha + beta * momage, c='k',
              label=f'ppvt = {alpha:.2f} + {beta:.2f} * momage')
     az.plot_hdi(momage, ppc['y_pred'], hdi_prob=0.97, color='gray')
     # Vizualizarea nu functioneaza, dar din moment ce Beta >0, in map_estimate, linia trasata are un comportament "cescator"
     # deci mamele cu o varsta mai inaintata au copii cu pptv mai mare
     model_educ=pm.Model()
     with model_educ:
         alpha = pm.Normal("alpha", mu=avg_ppvt, sigma=stdev_ppvt)
         beta = pm.Normal("beta", mu=avg_educ_cat, sigma=stdev_educ_cat)
         sigma = pm.HalfNormal("sigma", sigma=stdev_ppvt)
         # Expected value of outcome
         mu = pm.Deterministic('μ', alpha + beta * educ_cat)
         # Likelihood (sampling distribution) of observations
         Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=ppvt)
         idata_g = pm.sample(1000, tune=1000, return_inferencedata=True)
     map_estimate =pm.find_MAP(model=model_educ)
     print(map_estimate)
     ppc = pm.sample_posterior_predictive(idata_g, samples=100, model=model_educ)
     plt.plot(model_educ, alpha + beta * model_educ, c='k',
              label=f'ppvt = {alpha:.2f} + {beta:.2f} * educ_cat')
     az.plot_hdi(momage, ppc['ppvt_pred'], hdi_prob=0.97, color='gray')
     #Ca la punctul anterior, Beta=5, deci beta=0 in map_estimate, deci mamele cu categorie de educatie mai mare au copii cu ppvt mai mare
     #az.plot_posterior(idata_g)
     plt.show()
