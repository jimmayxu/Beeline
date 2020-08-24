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


pwd = '/mnt/PyCharmProjects/Beeline/outputs/SERGIO'



addon = ['_poissonloss.csv', '_mse_z-score.csv', '_mse_logtran.csv']

auroc_name = 'GSD-AUROC' + pd.Series(addon)
auprc_name = 'GSD-AUPRC' + pd.Series(addon)
epr_name = 'GSD-EPr' + pd.Series(addon)
loss = ['Poisson loss w/ counts', 'MSE loss w/ z-score', 'MSE loss w/ log trans']
import pandas as pd
import os
AUROC = pd.DataFrame(columns=loss)
AUPRC = pd.DataFrame(columns=loss)
EPR = pd.DataFrame(columns=loss)
for i, name in enumerate(loss):
    x = pd.read_csv(os.path.join(pwd, auroc_name[i]), index_col=0)
    AUROC[name] = pd.Series(x.loc['INVASE', :])
    x = pd.read_csv(os.path.join(pwd, auprc_name[i]), index_col=0)
    AUPRC[name] = pd.Series(x.loc['INVASE', :])
    x = pd.read_csv(os.path.join(pwd, epr_name[i]), index_col=0)
    EPR[name] = pd.Series(x.loc['INVASE', :])


import matplotlib.pyplot as plt
import numpy as np
out = ['AURPC_INVASE', 'AUROC_INVASE', 'EPR_INVASE']
for i, metric in enumerate([AUROC, AUPRC, EPR]):
    metric = metric[:-4]
    ax = metric.plot()
    x = np.arange(len(metric.index))
    ax.set_xticks(x)
    ax.set_xticklabels(metric.index)
    plt.xticks(rotation=90)
    plt.title(out[i])
    plt.tight_layout()
    plt.savefig(pwd + '/%s.png' %out[i])
    plt.show()