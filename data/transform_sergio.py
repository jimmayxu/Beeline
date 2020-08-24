%load_ext autoreload
%autoreload 2

import pandas as pd
from scipy.linalg import block_diag
import numpy as np

import os, sys
sys.path.append(os.getcwd())

nGenes = 450
nCells = 2500
number_bins = 9

from data import SergioDataset

sergio_params = {
    'number_genes': nGenes,
    'number_sc': nCells,
    'number_bins': number_bins,
    'input_filename_targets': 'steady-state_input_GRN%dG.txt' %nGenes,
    'input_filename_regs': 'steady-state_input_MRs9bin%dG.txt' %nGenes,
    'save_path': 'inputs/SERGIO',
}
SERGIO = SergioDataset(*sergio_params.values())

for n in range(5):
    SERGIO.simulate(n=n)
    X_ = SERGIO.add_noise(noise=['Poisson'])
    path = 'inputs/SERGIO/SERGIO-450-%d'%n
    os.makedirs(path, exist_ok=True)
    df = pd.DataFrame(X_.toarray(), columns=SERGIO.gene_names,
                      index='Cell' + SERGIO.cell_names.astype(str))
    df.T.to_csv('%s/ExpressionCounts.csv' %path)
    for mu in [3.0, 6.0]:
        params = {'libsize_mu': mu, 'libsize_scale': 2}
        X_ = SERGIO.add_noise(noise=['LibSize', 'Poisson'], **params)
        zeros = (X_==0).sum()/X_.shape[0]/X_.shape[1]
        df = pd.DataFrame(X_.toarray(), columns=SERGIO.gene_names,
                          index='Cell' + SERGIO.cell_names.astype(str))
        path = 'inputs/SERGIO/SERGIO-450-%d-%d' % (n, int(zeros*100))
        os.makedirs(path, exist_ok=True)
        df.T.to_csv('%s/ExpressionCounts.csv' %path)

        print(df.max().max())
        xx = pd.Series(X_.toarray().flatten()).replace(0, np.nan).dropna()
        print(xx.quantile(q=(.5, .6, .8, .9, .99)))