import random

import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
if __name__ == '__main__':
    az.style.use('arviz-darkgrid')
    dummy_data = np.loadtxt('date.csv')
    x_1 = dummy_data[:, 0]
    y_1 = dummy_data[:, 1]
    order = 2
    x_1p = np.vstack([x_1**i for i
    in range(1, order+1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True))
    x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()
    plt.scatter(x_1s[0], y_1s)
    plt.xlabel('x')
    plt.ylabel('y')
    with pm.Model() as model_l:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=10)
        ε = pm.HalfNormal('ε', 5)
        μ = α + β * x_1s[0]
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_l = pm.sample(2000, return_inferencedata=True)

    with pm.Model() as model_p:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=10, shape=order)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_p = pm.sample(2000, return_inferencedata=True)
    x_new = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)
    α_l_post = idata_l.posterior['α'].mean(("chain", "draw")).values
    β_l_post = idata_l.posterior['β'].mean(("chain", "draw")).values
    y_l_post = α_l_post + β_l_post * x_new
    plt.plot(x_new, y_l_post, 'C1', label='linear model')
    α_p_post = idata_p.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_p.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = α_p_post + np.dot(β_p_post, x_1s)
    plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label=f'model order {order}')
    order = 5
    x_1p = np.vstack([x_1 ** i for i
                      in range(1, order + 1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True))
    x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()
    with pm.Model() as model_p:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=10, shape=order)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_p = pm.sample(2000, return_inferencedata=True)
    α_p_post = idata_p.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_p.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = α_p_post + np.dot(β_p_post, x_1s)
    plt.plot(x_1s[0][idx], y_p_post[idx], 'C3', label=f'model order {order}') # order=5
    with pm.Model() as model_p:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=100, shape=order)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_p = pm.sample(2000, return_inferencedata=True)
    α_p_post = idata_p.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_p.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = α_p_post + np.dot(β_p_post, x_1s)
    plt.plot(x_1s[0][idx], y_p_post[idx], 'C5', label=f'model order {order} sd=100')
    with pm.Model() as model_p:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=np.array([10,0.1,0.1,0.1,0.1]), shape=order)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_p = pm.sample(2000, return_inferencedata=True)
    α_p_post = idata_p.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_p.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = α_p_post + np.dot(β_p_post, x_1s)
    plt.plot(x_1s[0][idx], y_p_post[idx], 'C7', label=f'model order {order} sd=array')
    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()
    plt.show() #nu mai compileaza dupa plt.show ar trebui sters sa mearga pana la capat
    #ex 2
    #-----
    #-----
    x_1 = stats.uniform.rvs(-3,3,size=500)
    y_1 = stats.uniform.rvs(-5,5, size=500)
    #apoi repetam exercitiul
    order = 2
    x_1p = np.vstack([x_1 ** i for i
                      in range(1, order + 1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True))
    x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()
    plt.scatter(x_1s[0], y_1s)
    plt.xlabel('x')
    plt.ylabel('y')
    with pm.Model() as model_l:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=10)
        ε = pm.HalfNormal('ε', 5)
        μ = α + β * x_1s[0]
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_l = pm.sample(2000, return_inferencedata=True)

    with pm.Model() as model_p:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=10, shape=order)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_p = pm.sample(2000, return_inferencedata=True)
    x_new = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)
    α_l_post = idata_l.posterior['α'].mean(("chain", "draw")).values
    β_l_post = idata_l.posterior['β'].mean(("chain", "draw")).values
    y_l_post = α_l_post + β_l_post * x_new
    plt.plot(x_new, y_l_post, 'C1', label='linear model')
    α_p_post = idata_p.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_p.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = α_p_post + np.dot(β_p_post, x_1s)
    plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label=f'model order {order}')
    order = 5
    x_1p = np.vstack([x_1 ** i for i
                      in range(1, order + 1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True))
    x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()
    with pm.Model() as model_p:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=10, shape=order)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_p = pm.sample(2000, return_inferencedata=True)
    α_p_post = idata_p.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_p.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = α_p_post + np.dot(β_p_post, x_1s)
    plt.plot(x_1s[0][idx], y_p_post[idx], 'C3', label=f'model order {order}')  # order=5
    with pm.Model() as model_p:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=100, shape=order)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_p = pm.sample(2000, return_inferencedata=True)
    α_p_post = idata_p.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_p.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = α_p_post + np.dot(β_p_post, x_1s)
    plt.plot(x_1s[0][idx], y_p_post[idx], 'C5', label=f'model order {order} sd=100')
    with pm.Model() as model_p:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=np.array([10, 0.1, 0.1, 0.1, 0.1]), shape=order)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_p = pm.sample(2000, return_inferencedata=True)
    α_p_post = idata_p.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_p.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = α_p_post + np.dot(β_p_post, x_1s)
    plt.plot(x_1s[0][idx], y_p_post[idx], 'C7', label=f'model order {order} sd=array')
    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()
    plt.show()
    #ex3
    #-----
    #-----
    x_1 = dummy_data[:, 0]
    y_1 = dummy_data[:, 1]
    order = 3
    x_1p = np.vstack([x_1 ** i for i
                      in range(1, order + 1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True))
    x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()
    with pm.Model() as model_c:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=10, shape=order)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_c = pm.sample(2000, return_inferencedata=True)
    α_p_post = idata_p.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_p.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    order = 2
    x_1p = np.vstack([x_1 ** i for i
                      in range(1, order + 1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True))
    x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()
    with pm.Model() as model_p:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=10, shape=order)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_p = pm.sample(2000, return_inferencedata=True)
    α_p_post = idata_p.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_p.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    waic_l = az.compare({'model_l':idata_l, 'model_p':idata_p, 'model_c':idata_c},method='BB-pseudo-BMA', ic="waic", scale="deviance")
    loo_l = az.compare({'model_l':idata_l, 'model_p':idata_p, 'model_c':idata_c},method='BB-pseudo-BMA', ic="loo", scale="deviance")
