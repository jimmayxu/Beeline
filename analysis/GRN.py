%load_ext autoreload
%autoreload 2

import pandas as pd
import os
import matplotlib.pyplot as plt
path = '/mnt/PyCharmProjects/Beeline/inputs/scRNA-Seq/Networks/human'
names = os.listdir(path)
for name in names:
    xx =pd.read_csv(os.path.join(path, name), header=0)
    ax = xx['Gene2'].value_counts().hist(bins=20)
    ax.set_xlabel('number of regulating TFs per target gene')
    ax.set_ylabel('frequency')
    ax.set_title(name[:-4])
    plt.savefig('%s.png' %os.path.join(path, name[:-4]))
    plt.show()

d = ['mHSC-L', 'mHSC-GM', 'mHSC-E']
path = '/mnt/PyCharmProjects/Beeline/inputs/scRNA-Seq/'

hESC_ExpressionData = pd.DataFrame()
for dd in d:
    p = os.path.join(path, dd, 'ExpressionData.csv')
    xx = pd.read_csv(p, index_col=0, header=0).T
    hESC_ExpressionData=hESC_ExpressionData.append(xx)
hESC_ExpressionData = hESC_ExpressionData.drop_duplicates(keep='first')

p = os.path.join(path, 'mHSC_full', 'ExpressionDataBeeline.csv')
hESC_ExpressionData.T.to_csv(p)




for dd in d:
    p = os.path.join(path, dd, 'ExpressionData.csv')
    xx = pd.read_csv(p, index_col=0, header=0).T
    ax = pd.Series(xx.values.flatten()).hist()
    ax.set_xlabel('normalised gene expression')
    ax.set_ylabel('frequency')
    ax.set_title(dd)
    plt.savefig(os.path.join(path, dd, 'expression_hist.png'))
    plt.show()


import pandas as pd
import os

network_file = 'Non-Specific-ChIP-seq-network'
ishuman = 'mouse'
folder_name = 'mHSC_full'
path = '/mnt/PyCharmProjects/Beeline/inputs/scRNA-Seq/'
p = os.path.join(path, folder_name, 'ExpressionData.csv')
ExpressionData = pd.read_csv(p, index_col=0, header=0).T
ExpressionData.columns = ExpressionData.columns.str.upper()
Gene_names = ExpressionData.columns

path_net = '/mnt/PyCharmProjects/Beeline/inputs/scRNA-Seq/Networks/%s/%s.csv' %(ishuman, network_file)
network = pd.read_csv(path_net, header=0)


tf_path = '/mnt/PyCharmProjects/Beeline/inputs/scRNA-Seq/%s-tfs.csv' %ishuman
known_TF_Genes = pd.read_csv(tf_path)['TF']
TF = Gene_names[Gene_names.isin(known_TF_Genes)]
isTF = pd.Series(network['Gene1'].unique()).isin(TF)
isTF.sum()/len(isTF)

TG = Gene_names[Gene_names.isin(network['Gene2'])]
#pd.Series(TG,name='TG').to_csv(os.path.join(path, folder_name, 'TGs_cell-type-specific-network.csv'))



len(network)
import numpy as np
f1 = pd.Series(network['Gene1']).isin(Gene_names)
f1.sum()/len(f1)
network = network[f1]
len(network)
f2 = pd.Series(network['Gene2']).isin(Gene_names)
f2.sum()/len(f2)
network = network[f2]
len(set(network['Gene1']))
len(set(network['Gene2']))
len(network)/len(set(network['Gene2']))/len(set(network['Gene1']))

network['Gene2'].value_counts()
network.to_csv(os.path.join(path, folder_name,  '%s_fit.csv'%network_file))

"""
network.to_csv(os.path.join(path, 'hESC',  'hESC-ChIP-seq-network_fit.csv'))
path_net = '/mnt/PyCharmProjects/Beeline/inputs/scRNA-Seq/Networks/%s/%s.csv' %(ishuman, network_file)
network = pd.read_csv(path_net, header=0)


new_path = '/mnt/PyCharmProjects/Beeline/inputs/scRNA-Seq/%s' %folder_name

p = os.path.join(path, folder_name, 'ExpressionDataBeeline.csv')
ExpressionData = pd.read_csv(p, index_col=0, header=0).T

Gene_names = ExpressionData.index
f1 = pd.Series(network['Gene1']).isin(Gene_names)
f1.sum()/len(f1)
network = network[f1]
len(network)
f2 = pd.Series(network['Gene2']).isin(Gene_names)
f2.sum()/len(f2)
network = network[f2]


tf_path = '/mnt/PyCharmProjects/Beeline/inputs/scRNA-Seq/mouse-tfs.csv'
known_TF_Genes = pd.read_csv(tf_path)['TF']
selected_gene = network['Gene2'].value_counts().index[:1000]
filter_gene2 = network['Gene2'].isin(selected_gene)
sum(filter_gene2)/len(filter_gene2)
sub_network = network[filter_gene2]
filter_gene1 = sub_network['Gene1'].isin(known_TF_Genes)
sum(filter_gene1)/len(filter_gene1)
sub_network = sub_network[filter_gene1]


TG_gene = sub_network['Gene2'].unique()
TF_gene = sub_network['Gene1'].unique()

len(sub_network)/len(TG_gene)/len(TF_gene)

ExpressionData = ExpressionData.T[pd.Index(set(TG_gene).union(set(TF_gene)))]
ExpressionData.T.to_csv(os.path.join(path, folder_name, 'ExpressionData.csv'))

pd.Series(TG_gene).to_csv(os.path.join(new_path, 'TGs.csv'))
pd.Series(TF_gene).to_csv(os.path.join(new_path, 'TFs.csv'))

#pd.read_csv(os.path.join(new_path, 'TG1000.csv'), index_col=0, squeeze=True)
"""


(ExpressionData==0).sum().sum()/ExpressionData.shape[1]/ExpressionData.shape[0]




XX = pd.read_csv('/mnt/PyCharmProjects/Beeline/inputs/SERGIO100/refNetwork.csv')
pd.Series(np.unique(XX['Gene1']), name = 'TF').to_csv('/mnt/PyCharmProjects/Beeline/inputs/SERGIO100/tfs.csv')
xx = pd.read_csv('/mnt/PyCharmProjects/Beeline/inputs/SERGIO100/tfs.csv')['TF']
import pandas as pd

name = 'mHSC_all'
xx = pd.read_csv('/mnt/PyCharmProjects/Beeline/inputs/scRNA-Seq/%s/ExpressionDataBeeline.csv'%name, index_col=0)
(xx==0).sum().sum()/xx.shape[0]/xx.shape[1]
ax = pd.Series(xx.values.flatten()).hist(bins=20)
ax.set_xlabel('normalised expression')
ax.set_title(name)
import matplotlib.pyplot as plt
plt.show()
aa = pd.read_csv('/mnt/PyCharmProjects/Beeline/inputs/scRNA-Seq/mouse-tfs.csv')['TF']
pd.Series(xx.T[xx.columns.isin(aa)].values.flatten()).hist(bins=20)
plt.show()


xx = pd.read_csv('/mnt/PyCharmProjects/Beeline/inputs/scRNA-Seq/mHSC_full/STRING-network_fit.csv')
len(set(xx['Gene2']))
len(set(xx['Gene1']))
len(xx)/len(set(xx['Gene2']))/len(set(xx['Gene1']))