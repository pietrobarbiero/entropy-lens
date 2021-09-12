import os
import pandas as pd

df = pd.read_csv('results_aware_vdem_l_1e-06_tau_6.csv')

meandf = df.groupby(['tau', 'lambda']).mean()
semdf = df.groupby(['tau', 'lambda']).sem()

for row in range(len(meandf)):
    rowid = meandf.index[row]
    print(f'| ${rowid}$ | ${meandf["model_accuracy"][rowid]*100:.02f} \pm {semdf["model_accuracy"][rowid]*100:.02f}$ |  ${meandf["explanation_accuracy"][rowid]*100:.02f} \pm {semdf["explanation_accuracy"][rowid]*100:.02f}$ |  ${meandf["explanation_complexity"][rowid]:.02f} \pm {semdf["explanation_complexity"][rowid]:.02f}$ |')

print()