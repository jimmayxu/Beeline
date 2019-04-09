import os
import pandas as pd
from pathlib import Path
import numpy as np

def generateInputs(RunnerObj):
    '''
    Function to generate desired inputs for SCODE.
    If the folder/files under RunnerObj.datadir exist, 
    this function will not do anything.
    '''
    if not RunnerObj.inputDir.joinpath("SINCERITIES").exists():
        print("Input folder for SINCERITIES does not exist, creating input folder...")
        RunnerObj.inputDir.joinpath("SINCERITIES").mkdir(exist_ok = False)
        
    if not RunnerObj.inputDir.joinpath("SINCERITIES/ExpressionData.csv").exists():
        ExpressionData = pd.read_csv(RunnerObj.inputDir.joinpath('ExpressionData.csv'),
                                     header = 0, index_col = 0)
        newExpressionData = ExpressionData.T.copy()
        PTData = pd.read_csv(RunnerObj.inputDir.joinpath('PseudoTime.csv'),
                             header = 0, index_col = 0)
        newExpressionData['Time'] = PTData['Time']
        newExpressionData.to_csv(RunnerObj.inputDir.joinpath("SINCERITIES/ExpressionData.csv"),
                             sep = ',', header  = True, index = False)
    
def run(RunnerObj):
    '''
    Function to run SINCERITIES algorithm
    '''
    inputPath = "data/" + str(RunnerObj.inputDir).split("ModelEval/")[1] + \
                    "/SINCERITIES/ExpressionData.csv"
    
    # make output dirs if they do not exist:
    outDir = "outputs/"+str(RunnerObj.inputDir).split("inputs/")[1]+"/SINCERITIES/"
    os.makedirs(outDir, exist_ok = True)
    
    outPath = "data/" + str(outDir) + 'outFile.txt'
    cmdToRun = ' '.join(['docker run --rm -v ~/ModelEval:/SINCERITIES/data/ sincerities:base /bin/sh -c \"Rscript MAIN.R', 
                         inputPath, outPath, '\"'])
    print(cmdToRun)
    os.system(cmdToRun)



def parseOutput(RunnerObj):
    '''
    Function to parse outputs from SCODE.
    '''
    # Quit if output directory does not exist
    outDir = "outputs/"+str(RunnerObj.inputDir).split("inputs/")[1]+"/SINCERITIES/"
    if not Path(outDir).exists():
        raise FileNotFoundError()
        
    # Read output
    OutDF = pd.read_csv(outDir+'outFile.txt', sep = ',', header = 0)
    
    outFile = open(outDir + 'rankedEdges.csv','w')
    outFile.write('Gene1'+'\t'+'Gene2'+'\t'+'EdgeWeight'+'\n')

    for idx, row in OutDF.iterrows():
        outFile.write('\t'.join([row['SourceGENES'],row['TargetGENES'],str(row['Interaction'])])+'\n')
    outFile.close()
    