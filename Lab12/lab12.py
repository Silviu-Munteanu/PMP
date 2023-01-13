import arviz as az
import pymc3 as pm
import matplotlib.pyplot as plt
if __name__=="__main__":
    centered=az.load_arviz_data("centered_eight")
    uncentered=az.load_arviz_data("non_centered_eight")
    az.plot_trace(centered, divergences='top', compact=False)
    az.plot_trace(uncentered, compact=False)      #in  cazul uncentered nu pare sa se blocheze niciun chain, iar in celalat caz se pot observa chainuri blocate
    print(az.ess(centered))                      #in ultimul rand din tabel
    print("==================")
    print(az.ess(uncentered))
    az.plot_posterior(centered)
    az.plot_posterior(uncentered)
    print("==================")
    print("Centered",az.rhat(centered)) #pt centered mu 1.027 tau 1.071
    print("==================")
    print("Uncentered",az.rhat(uncentered)) #pt uncentered  mu  1.0 tau 1.001
    az.plot_autocorr(centered)
    az.plot_autocorr(uncentered) #pt centered avem valori mai mari decat pentru uncentered in grafice
    print(centered.sample_stats.diverging.sum(),uncentered.sample_stats.diverging.sum()) # 43 centered, 2 uncentered
    az.plot_parallel(centered)
    az.plot_parallel(uncentered) #ambele se concentreaza in intervalul 0-10 pe axa Oy
    plt.show()
