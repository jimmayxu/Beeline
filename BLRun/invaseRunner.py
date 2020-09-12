import os
import pandas as pd
from pathlib import Path
import scipy.sparse as sp_sparse
from Algorithms.INVASE import INVASE
import numpy as np
from sklearn import preprocessing
import tensorflow as tf

def generateInputs(RunnerObj):
    '''
    Function to generate desired inputs for INVASE.
    If the folder/files under RunnerObj.datadir exist, 
    this function will not do anything.
    '''

    if not RunnerObj.inputDir.joinpath("INVASE").exists():
        print("Input folder for INVASE does not exist, creating input folder...")
        RunnerObj.inputDir.joinpath("INVASE").mkdir(exist_ok=False)
    """
    if not RunnerObj.inputDir.joinpath("INVASE/ExpressionData.csv").exists():
        ExpressionData = pd.read_csv(RunnerObj.inputDir.joinpath(RunnerObj.exprData),
                                     header = 0, index_col = 0)

        # Write .csv file
        ExpressionData.T.to_csv(RunnerObj.inputDir.joinpath("INVASE/ExpressionData.csv"),
                             sep = '\t', header  = True, index = True)
    """
    #if not RunnerObj.inputDir.joinpath("INVASE/ExpressionData.csv").exists():
    ExpressionCounts = pd.read_csv(RunnerObj.inputDir.joinpath('ExpressionCounts.csv'),
                                 header=0, index_col=0)
    ExpressionData = ExpressionCounts.copy()
    ExpressionData = pd.DataFrame(preprocessing.scale(ExpressionData), columns=ExpressionData.columns, index=ExpressionData.index)

    # Write .csv file
    ExpressionData.T.to_csv(RunnerObj.inputDir.joinpath("INVASE/ExpressionData.csv"),
                            sep='\t', header=True, index=True)
    """
    #if not RunnerObj.inputDir.joinpath("INVASE/ExpressionData.csv").exists():
    ExpressionCounts = pd.read_csv(RunnerObj.inputDir.joinpath('ExpressionCounts.csv'),
                                 header=0, index_col=0)
    Exp = ExpressionCounts.copy()
    ExpressionData = np.log(Exp + 1)
    # Write .csv file
    ExpressionData.T.astype(int).to_csv(RunnerObj.inputDir.joinpath("INVASE/ExpressionData.csv"),
                            sep='\t', header=True, index=True)
    """
    """
    #use BEELINE provided dataset 
    if not RunnerObj.inputDir.joinpath("INVASE/ExpressionData.csv").exists():
        ExpressionData = pd.read_csv(RunnerObj.inputDir.joinpath(RunnerObj.exprData),
                                     header=0, index_col=0)
        ExpressionCounts = pd.DataFrame(np.random.poisson(ExpressionData), columns=ExpressionData.columns, index=ExpressionData.index)
        Exp = ExpressionCounts.copy()
        # Write .csv file
        Exp.T.astype(int).to_csv(RunnerObj.inputDir.joinpath("INVASE/ExpressionData.csv"),
                                sep='\t', header=True, index=True)
        #ExpressionData.to_csv(RunnerObj.inputDir.joinpath(RunnerObj.exprData))
    """

def run(RunnerObj):
    '''
    Function to run INVASE algorithm
    '''
    inputPath = str(RunnerObj.inputDir) + "/INVASE/ExpressionData.csv"
    # make output dirs if they do not exist:
    outDir = "outputs/"+str(RunnerObj.inputDir).split("inputs/")[1]+"/INVASE/"
    os.makedirs(outDir, exist_ok=True)
    inDF = pd.read_csv(inputPath, sep='\t', index_col=0, header=0)
    gene_names = inDF.columns.str.upper()

    TF_file = RunnerObj.params.get('TF_file', None)
    TG_file = RunnerObj.params.get('TG_file', None)
    loss_type = RunnerObj.params.get('loss_type', None)
    if TF_file is not None:
        TF_Genes = pd.read_csv(os.path.join(str(RunnerObj.inputDir), TF_file))['TF'].str.upper()
        TF_Genes = TF_Genes[TF_Genes.isin(gene_names)]
    else:
        TF_Genes = list(gene_names)

    if TG_file is not None:
        target_Genes = pd.read_csv(os.path.join(str(RunnerObj.inputDir), TG_file))['TG'].str.upper()
    else:
        target_Genes = list(gene_names)
    """
    trueEdges = pd.read_csv(os.path.join(str(RunnerObj.inputDir), RunnerObj.trueEdges))
    gene_network = set(trueEdges.values.flatten())
    gene_temp = gene_network - set(TF_Genes)
    target_Genes = list(set(gene_names).intersection(gene_temp))
    """

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

    iteration = int(RunnerObj.params['iteration'])
    batch_size = int(RunnerObj.params['batch_size'])
    method = INVASE(iteration=iteration, batch_size=batch_size, loss_type=loss_type, **kwargs)
    #method = INVASE(epochs=epochs, batch_size=batch_size, loss_type='poisson', **kwargs)

    X_ = sp_sparse.csc_matrix(inDF.values)
    TF2Gene_Prob, TF2Gene_Binary = method.fit(X_)
    outPathBinary = str(outDir) + 'outFile_binary.txt'
    outPathProb = str(outDir) + 'outFile_prob.txt'
    TF2Gene_Prob.to_csv(outPathProb, index=False, sep='\t')
    TF2Gene_Binary.to_csv(outPathBinary, index=False, sep='\t')

def parseOutput(RunnerObj):
    '''
    Function to parse outputs from INVASE.
    '''
    # Quit if output directory does not exist
    outfile = 'outFile_prob.txt'
    outDir = "outputs/" + str(RunnerObj.inputDir).split("inputs/")[1] + "/INVASE/"

    if not Path(outDir + outfile).exists():
        print(outDir + outfile + 'does not exist, skipping...')
        return
    # Read output
    OutDF = pd.read_csv(outDir + outfile, sep='\t', header=0)
    outFile = open(outDir + 'rankedEdges.csv', 'w')
    outFile.write('Gene1' + '\t' + 'Gene2' + '\t' + 'EdgeWeight' + '\n')
    for idx, row in OutDF.iterrows():
        outFile.write('\t'.join([row['TF'], row['target'], str(row[OutDF.columns[-1]])]) + '\n')
    outFile.close()

    outfile = 'outFile_binary.txt'
    OutDF = pd.read_csv(outDir + outfile, sep='\t', header=0)
    outFile = open(outDir + 'rankedEdges2.csv', 'w')
    outFile.write('Gene1' + '\t' + 'Gene2' + '\t' + 'EdgeWeight' + '\n')
    for idx, row in OutDF.iterrows():
        outFile.write('\t'.join([row['TF'], row['target'], str(row[OutDF.columns[-1]])]) + '\n')
    outFile.close()

