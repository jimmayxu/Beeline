import os
import pandas as pd
from pathlib import Path
import scipy.sparse as sp_sparse
from Algorithms.INVASE import INVASE
import numpy as np
def generateInputs(RunnerObj):
    '''
    Function to generate desired inputs for INVASE.
    If the folder/files under RunnerObj.datadir exist, 
    this function will not do anything.
    '''
    if not RunnerObj.inputDir.joinpath("INVASE").exists():
        print("Input folder for INVASE does not exist, creating input folder...")
        RunnerObj.inputDir.joinpath("INVASE").mkdir(exist_ok=False)

    if not RunnerObj.inputDir.joinpath("INVASE/ExpressionData.csv").exists():
        ExpressionData = pd.read_csv(RunnerObj.inputDir.joinpath(RunnerObj.exprData),
                                     header=0, index_col=0)

        # Write .csv file
        ExpressionData.T.to_csv(RunnerObj.inputDir.joinpath("INVASE/ExpressionData.csv"),
                                sep='\t', header=True, index=True)


def run(RunnerObj):
    '''
    Function to run INVASE algorithm
    '''
    inputPath = str(RunnerObj.inputDir) + "/INVASE/ExpressionData.csv"
    # make output dirs if they do not exist:
    outDir = str(RunnerObj.inputDir) + "/INVASE/"
    os.makedirs(outDir, exist_ok=True)
    outPath = str(outDir) + 'outFile.txt'
    inDF = pd.read_csv(inputPath, sep='\t', index_col=0, header=0)

    gene_names = inDF.columns
    known_TF_Genes = np.loadtxt('%s/TF_names.txt' % str(RunnerObj.inputDir), dtype='str').tolist()
    TF_Genes = list(gene_names.intersection(known_TF_Genes))
    target_Genes = list(set(gene_names) - set(TF_Genes))

    kwargs = dict()
    kwargs.update(
        {"TF_Genes": TF_Genes}
    )
    kwargs.update(
        {'gene_names': gene_names}
    )
    kwargs.update(
        {'target_Genes': target_Genes}
    )

    epochs = int(RunnerObj.params['epochs'])
    batch_size = int(RunnerObj.params['batch_size'])
    method = INVASE(folder_name='trial', epochs=epochs, batch_size=batch_size, **kwargs)

    X_ = sp_sparse.csc_matrix(inDF.values)
    network = method.fit(X_)
    network.to_csv(outPath, index=False, sep='\t')

def parseOutput(RunnerObj):
    '''
    Function to parse outputs from INVASE.
    '''
    # Quit if output directory does not exist
    outDir = "outputs/" + str(RunnerObj.inputDir).split("inputs/")[1] + "/INVASE/"

    if not Path(outDir + 'outFile.txt').exists():
        print(outDir + 'outFile.txt' + 'does not exist, skipping...')
        return
    # Read output
    OutDF = pd.read_csv(outDir + 'outFile.txt', sep='\t', header=0)

    outFile = open(outDir + 'rankedEdges.csv', 'w')
    outFile.write('Gene1' + '\t' + 'Gene2' + '\t' + 'EdgeWeight' + '\n')

    for idx, row in OutDF.iterrows():
        outFile.write('\t'.join([row['TF'], row['target'], str(row['importance'])]) + '\n')
    outFile.close()
