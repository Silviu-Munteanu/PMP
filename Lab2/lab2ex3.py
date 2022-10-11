import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import math
import random
import arviz as az
coin1s=np.random.binomial(10, 0.5, 100)
coin2s=np.random.binomial(10, 0.3, 100)
bb=np.zeros(100)
bs=np.zeros(100)
sb=np.zeros(100)
ss=np.zeros(100)
coin1b=[]
coin2b=[]
for i in range(100):
    coin1b.append(10-coin1s[i])
    coin2b.append(10-coin2s[i])
az.plot_posterior({"moneda 1 ban": coin1b,"moneda 1 stema":coin1s,"moneda 2 ban":coin2b,"moneda 2 stema":coin2s})
for i in range(100):
        while coin1b[i]:
            if random.random()<coin2s[i]/(coin2s[i]+coin2b[i]):       #sansa ca banul monedei 1 sa fi picat cu stema monedei 2
                coin2s[i]-=1
                bs[i]+=1
            else:
                coin2b[i] -= 1                                        # cu banul monedei 2
                bb[i]+=1
            coin1b[i]-=1
        while coin1s[i]:
            if random.random()<coin2s[i]/(coin2s[i]+coin2b[i]):      #stema 1 cu stema 2
                coin2s[i]-=1
                ss[i]+=1
            else:
                coin2b[i]-=1                                         #stema 1 cu banul 2
                sb[i]+=1
            coin1s[i]-=1
az.plot_posterior({"bb":bb,"sb":sb,"bs":bs,"ss":ss})
print(bb,sb,bs,ss)
plt.show()