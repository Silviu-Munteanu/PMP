import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
def posterior_grid(grid_points=50, heads=6, tails=9):
    grid = np.linspace(0, 1, grid_points)
    prior = np.repeat(1/grid_points, grid_points)
    likelihood = stats.binom.pmf(heads, heads+tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior
def posterior_grid_curs(grid_points=50, heads=6, tails=9):
    grid = np.linspace(0, 1, grid_points)
    prior =(grid<= 0.5).astype(int)
    likelihood = stats.binom.pmf(heads, heads+tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return posterior
def estimate_pi(n):
    x, y = np.random.uniform(-1, 1, size=(2, n))
    inside = (x ** 2 + y ** 2) <= 1
    pi = inside.sum() * 4 / n
    error = abs((pi - np.pi) / pi) * 100
    return error
def metropolis(func, draws=10000):
    trace = np.zeros(draws)
    old_x = 0.5 # func.mean()
    old_prob = func.pdf(old_x)
    delta = np.random.normal(0, 0.5, draws)
    for i in range(draws):
        new_x = old_x + delta[i]
        new_prob = func.pdf(new_x)
        acceptance = new_prob / old_prob
        if acceptance >= np.random.random():
            trace[i] = new_x
            old_x = new_x
            old_prob = new_prob
        else:
            trace[i] = old_x
    return trace
print(posterior_grid(100,20,20))
print("============================")
l100=[]
l1000=[]
l10000=[]
for i in range(5000):
    l100.append(estimate_pi(100))
    l1000.append(estimate_pi(1000))
    l10000.append(estimate_pi(10000))
print(np.mean(l100), np.std(l100))
print(np.mean(l1000), np.std(l1000))
print(np.mean(l10000), np.std(l10000))
plt.errorbar(l100,l1000)
plt.show()
plt.errorbar(l1000,l10000)
plt.show()
print(np.mean(l100)/np.mean(l1000))
print(np.mean(l1000)/np.mean(l10000))
#eroare(n1) = eroare(n2) * sqrt(n1 / n2)
print("============================")

beta_params=[(1, 1), (20, 20), (1, 4)]
for i in range(3):
    func = stats.beta(beta_params[i][0],beta_params[i][1])
    trace = metropolis(func=func)
    x = np.linspace(0, 1, 100)
    plt.plot(trace)
    plt.plot(posterior_grid_curs(100,beta_params[i][0],beta_params[i][1]))
    print(trace)
    print(posterior_grid_curs(100,beta_params[i][0],beta_params[i][1]))
    plt.show()
