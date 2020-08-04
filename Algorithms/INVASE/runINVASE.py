from optparse import OptionParser
import sys
import pandas as pd
from Algorithms import INVASE

def parseArgs(args):
    parser = OptionParser()

    parser.add_option('', '--algo', type='str',
                      help='Algorithm to run. Can either by GENIE3 or GRNBoost2')

    parser.add_option('', '--inFile', type='str',
                      help='Path to input tab-separated expression SamplesxGenes file')

    parser.add_option('', '--outFile', type='str',
                      help='File where the output network is stored')

    (opts, args) = parser.parse_args(args)

    return opts, args


def main(args):
    opts, args = parseArgs(args)
    inDF = pd.read_csv(opts.inFile, sep='\t', index_col=0, header=0)


    method = INVASE(folder_name='trial', epochs=100, batch_size=1000, **kwargs)
    network = method.fit(inDF.values)
    network.to_csv(opts.outFile, index=False, sep='\t')



if __name__ == "__main__":
    main(sys.argv)
