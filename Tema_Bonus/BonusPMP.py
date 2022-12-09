import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import arviz as az
from scipy import stats

samples=5000
clienti=stats.poisson.rvs(20,size=samples)
plasare=stats.truncnorm.rvs(-2,np.inf,1,0.5,10000) #fara valori negative, adica maxim 2 deviatii standard la stanga
gatit=stats.expon.rvs(0,2,10000)
mancat=stats.truncnorm.rvs(-5,np.inf,10,2,10000)
for t in range(1, 10000): #incercam crescator nr de case pana se iese cu break
    ok=0
    sub15 = 0
    for i in range(samples):
        cln = clienti[i]  # cati clienti trebuie serviti
        timpi = []
        case_plasat=np.zeros(t)
        timp_la_coada=0
        peste15=0
        while cln:
            gasit=0
            for k in range(len(case_plasat)):
                if case_plasat[k]==0:
                    case_plasat[k]=stats.truncnorm.rvs(-2,np.inf,1,0.5)
                    gasit=1
                    timp=case_plasat[k] + stats.expon.rvs(0,2) + timp_la_coada #plasat + gatit + stat la coada
                    if timp>=15:
                        peste15=1
                    break
            if gasit==0:
                timp_la_coada+=min(case_plasat)     #pana se elibereaza urmatoarea casa
                case_plasat-=min(case_plasat)
            if gasit == 1:
                cln-=1
        if peste15 == 1:
            continue
        sub15+=1
    print(sub15/samples)
    if sub15>=0.95 * samples: # evident cu sampleuri nu se gaseste un nr exact, cu mai multe sampleuri ar trebui sa se apropie de realitate
        print("Nr minim case:",t)
        break
for t in range(1, 10000):
    ok=0
    sub15 = 0
    for i in range(samples):
        cln = clienti[i]
        timpi = []
        statii=np.zeros(t)
        timp_la_coada=0
        peste15=0
        while cln:
            gasit=0
            for k in range(len(statii)):
                if statii[k]==0:
                    statii[k]=stats.expon.rvs(0,2)
                    gasit=1
                    timp=statii[k] + stats.truncnorm.rvs(-2,np.inf,1,0.5) + timp_la_coada
                    if timp>=15:
                        peste15=1
                    break
            if gasit==0:
                timp_la_coada+=min(statii)
                statii-=min(statii)
            if gasit == 1:
                cln-=1
        if peste15 == 1:
            continue
        sub15+=1
    print(sub15/samples)
    if sub15>=0.95 * samples:
        print("Nr minim statii:",t)
        break
for t in range(1, 10000):
    ok=0
    sub15 = 0
    for i in range(samples):
        cln = clienti[i]
        timpi = []
        mese=np.zeros(t)
        timp_la_coada=0
        peste15=0
        while cln:
            gasit=0
            for k in range(len(mese)):
                if mese[k]==0:
                    mese[k]=stats.truncnorm.rvs(-5,np.inf,10,2)
                    gasit=1
                    timp=mese[k]  + timp_la_coada
                    if timp>=15:
                        peste15=1
                    break
            if gasit==0:
               timp_la_coada+=min(mese)
               mese-=min(mese)
            if gasit == 1:
                cln-=1
        if peste15 == 1:
            continue
        sub15+=1
    print(sub15/samples)
    if sub15>=0.95 * samples:
        print("Nr minim mese:",t)
        break
