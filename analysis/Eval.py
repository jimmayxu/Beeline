pwd = '/mnt/PyCharmProjects/Beeline/outputs/Curated/GSD/GSD-AUPRC.csv'

import pandas as pd
xx = pd.read_csv(pwd, index_col=0)

import matplotlib.pyplot as plt
import numpy as np
ax = xx.T.plot()
x = np.arange(len(xx.columns))
ax.set_xticks(x)
ax.set_xticklabels(xx.columns)
plt.xticks(rotation=90)
plt.savefig
plt.show()
