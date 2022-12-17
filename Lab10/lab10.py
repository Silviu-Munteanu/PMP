import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import pymc3 as pm
from theano import tensor as tt
if __name__ == '__main__':
    clusters = 3
    n_cluster = [200, 100,200]
    n_total = sum(n_cluster)
    means = [0, 5, 10]
    std_devs = [2, 1,2.5]                   #le-am ales sa se si vada destul de usor ca sunt 3
    mix = np.random.normal(np.repeat(means, n_cluster),
    np.repeat(std_devs, n_cluster))
    az.plot_kde(np.array(mix))
    clusters = 2
    with pm.Model() as model_c2:
        p = pm.Dirichlet('p', a=np.ones(clusters))
        means = pm.Normal('means', mu=np.array([0, 5]) * mix.mean(), sd=10, shape=clusters)
        sd = pm.HalfNormal('sd', sd=10)
        order_means = pm.Potential('order_means', tt.switch(means[1]-means[0] < 0, -np.inf, 0))
        y = pm.NormalMixture('y', w=p, mu=means, sd=sd, observed=mix)
        data_2 = pm.sample(1000,return_inferencedata=True)
    clusters = 3
    with pm.Model() as model_c3:
        p = pm.Dirichlet('p', a=np.ones(clusters))
        means = pm.Normal('means', mu=np.array([0, 2.5,5]) * mix.mean(), sd=10, shape=clusters)
        sd = pm.HalfNormal('sd', sd=10)
        order_means = pm.Potential('order_means', tt.switch(means[1] - means[0] < 0, -np.inf, 0))
        y = pm.NormalMixture('y', w=p, mu=means, sd=sd, observed=mix)
        data_3 = pm.sample(1000,return_inferencedata=True)
    clusters = 4
    with pm.Model() as model_c4:
        p = pm.Dirichlet('p', a=np.ones(clusters))
        means = pm.Normal('means', mu=np.array([0, 2,4,6]) * mix.mean(), sd=10, shape=clusters)
        sd = pm.HalfNormal('sd', sd=10)
        order_means = pm.Potential('order_means', tt.switch(means[1] - means[0] < 0, -np.inf, 0))
        y = pm.NormalMixture('y', w=p, mu=means, sd=sd, observed=mix)
        data_4 = pm.sample(1000,return_inferencedata=True)
        waic_l = az.compare({'2 distributions': data_2, '3 distributions': data_3, '4 distributions': data_4}, method='BB-pseudo-BMA',
                            ic="waic", scale="deviance")
        loo_l = az.compare({'2 distributions': data_2, '3 distributions': data_3, '4 distributions': data_4}, method='BB-pseudo-BMA',
                           ic="loo", scale="deviance")
        print(waic_l)
        print(loo_l)
        #Pentru 100 samples si 100 tune modelul cu 4 distributii a avut cele mai bune rezultate atat la waic cat si la loo
    plt.show()
