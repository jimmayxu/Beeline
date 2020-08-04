%load_ext autoreload
%autoreload 2

import yaml
import argparse
import os
import BLRun as br
yaml.warnings({'YAMLLoadWarning': False})

config_file = 'config-files/config.yaml'

with open(config_file, 'r') as conf:
    evaluation = br.ConfigParser.parse(conf)
print(evaluation)
print('Evaluation started')

idx = 0 #PIDC
evaluation.runners[idx].generateInputs()

evaluation.runners[idx].run()

from pathlib import Path

def run(RunnerObj):
    '''
    Function to run PIDC algorithm
    '''
    inputPath = 'inputs/example/GSD/PIDC/ExpressionData.csv'
    # make output dirs if they do not exist:
    outDir = 'outputs/example/GSD/PIDC/'
    os.makedirs(outDir, exist_ok=True)

    outPath = str(outDir) + 'outFile.txt'
    cmdToRun = ' '.join(['docker run --rm -v', str(Path.cwd()) + ':/data pidc:base /bin/sh -c \"time -v -o',
                        str(outDir) + 'time.txt', 'julia runPIDC.jl',
                         inputPath, outPath, '\"'])
    print(cmdToRun)
    os.system(cmdToRun)


import pandas as pd
inPathData = 'data/expr_2000sc_9bins100G.csv'
X_pd = pd.read_csv(inPathData)
X = X_pd.T
outPathData= 'data/toy.csv'
X.to_csv(outPathData, index=True, sep=' ')


cmdToRun = 'julia runPIDC.jl /Users/zx3/PycharmProjects/Beeline/data/toy.csv /Users/zx3/PycharmProjects/Beeline/outputs/example/GSD/PIDC/outFile.txt'