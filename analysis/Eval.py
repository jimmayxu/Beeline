%load_ext autoreload
%autoreload 2

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

import pandas as pd
import os

addon = ['_poissonloss.csv', '_mse_z-score.csv', '_mse_logtran.csv']

auroc_name = 'GSD-AUROC' + pd.Series(addon)
auprc_name = 'GSD-AUPRC' + pd.Series(addon)
epr_name = 'GSD-EPr' + pd.Series(addon)
loss = ['Poisson loss w/ counts', 'MSE loss w/ z-score', 'MSE loss w/ log trans']

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
out = ['AUROC_INVASE', 'AUPRC_INVASE', 'EPR_INVASE']
for i, metric in enumerate([AUROC, AUPRC, EPR]):
    ax = metric.plot()
    x = np.arange(len(metric.index))
    ax.set_xticks(x)
    ax.set_xticklabels(metric.index)
    plt.xticks(rotation=90)
    plt.title(out[i])
    plt.tight_layout()
    plt.savefig(pwd + '/%s.png' %out[i])
    plt.show()





# bins the genes by the number of regulating TFs, evaluate the interactions of each bins of genes inferred by INVASE.


import pandas as pd

network_file = 'mHSC-ChIP-seq-network1000.csv'
folder_name = 'mHSC_all'
#network_file = 'STRING-network_fit.csv'
#folder_name = 'hESC'
path_net = '/mnt/PyCharmProjects/Beeline/inputs/scRNA-Seq/%s/%s' %(folder_name, network_file)

network = pd.read_csv(path_net, header=0, index_col = 0)
len(set(network['Gene2']))
import matplotlib.pyplot as plt
network['Gene2'].value_counts().hist()
plt.show()
vc = network['Gene2'].value_counts()
quan = vc.quantile([0, 0.1, 0.5, 0.9, 1]).values


range(len(vc))
methods = ['INVASE', 'GRNBOOST2']
AUPR = pd.DataFrame()
from BLEval.computeAUC import computeScores_new
for method in methods:
    out_path = 'outputs/scRNA-Seq/%s/%s/rankedEdges.csv' %(folder_name, method)
    predDF = pd.read_csv(out_path, sep = '\t', header =  0, index_col = None)
    _,_,_,_, aupr1, _ = computeScores_new(network, predDF, edges_TFTG=True)
    AUPR.loc['all', method] = aupr1

for method in methods:
    out_path = 'outputs/scRNA-Seq/%s/%s/rankedEdges.csv' %(folder_name, method)
    predDF = pd.read_csv(out_path, sep = '\t', header =  0, index_col = None)
    for q in range(len(quan)-1):
        S = vc.between(quan[q], quan[q+1])
        TG = vc.index[S]
        subnet = network[network['Gene2'].isin(TG)]
        subpredDF = predDF[predDF['Gene2'].isin(TG)]
        prec2,_,_,_, aupr2, _ = computeScores_new(subnet, subpredDF, edges_TFTG=True)
        AUPR.loc['%d-%d'%(quan[q], quan[q+1]), method] =  aupr2



out_path = 'outputs/scRNA-Seq/mHSC_full/GRNBOOST2/rankedEdges.csv'
out_path = 'outputs/scRNA-Seq/mHSC_full/GRNBOOST2/rankedEdges.csv'
predDF1 = pd.read_csv(out_path, sep = '\t', header =  0, index_col = None)
predDF1['Gene2'].value_counts()
predDF1.sort_values(by=['EdgeWeight', 'Gene1'], ascending=False)[:20]
np.log(predDF1['EdgeWeight']).hist()
import matplotlib.pyplot as plt
plt.show()

out_path = 'outputs/scRNA-Seq/mHSC_full/INVASE/rankedEdges.csv'
predDF2 = pd.read_csv(out_path, sep = '\t', header =  0, index_col = None)
predDF2['EdgeWeight'].hist()
import matplotlib.pyplot as plt
plt.show()

predDF2 = predDF2[predDF2['EdgeWeight']>0.5]
predDF2['Gene2'].value_counts()
predDF2.sort_values(by=['EdgeWeight', 'Gene1'], ascending=False)[:20]

predDF2.sort_values(by='EdgeWeight')
predDF2['Gene2'].value_counts()



out_path = 'outputs/SERGIO/SERGIO-450-1-86/PIDC/rankedEdges.csv'
predDF = pd.read_csv(out_path, sep='\t', header=0, index_col=None)