import numpy as np
import pandas as pd
import os, time
from tqdm import tqdm


class Feature_Selection_models:
    def __init__(
            self,
            method_name,
            **kwargs
    ):
        self.method_name = method_name

        self.i = kwargs.get('i', '')
        self.CellLabel = kwargs.get('CellLabel', None)

        self.target_Genes = kwargs.get('target_Genes', None)
        self.TF_Genes = kwargs.get('TF_Genes', None)
        self.gene_names = kwargs.get('gene_names', None)
        assert self.TF_Genes is not None
        assert all(pd.Series(self.TF_Genes).isin(self.gene_names))

        self.kwargs = kwargs

    def fit(self, X_, config):
        #assert pd.Series(self.target_Genes).isin(self.gene_names).all()
        index = np.where(self.gene_names.isin(self.TF_Genes))[0]
        target_index = np.where(self.gene_names.isin(self.target_Genes))[0]

        self.x_train = X_[:, index].toarray()
        self.test_idx = None

        print('\nTraining for %s' % self.method_name)
        self.bin_Prob = dict()
        self.bin_Binary = dict()
        self.TF2Gene_Prob = pd.DataFrame(columns=["TF", "target", "probability"])
        self.TF2Gene_Binary = pd.DataFrame(columns=["TF", "target", "vote"])
        i = 0
        for target_Gene in tqdm(
                self.target_Genes,
                desc=("Training %s model for each target gene" % (self.method_name))
        ):
            if target_Gene in self.TF_Genes:
                i = pd.Series(self.TF_Genes).isin([target_Gene])
                x_train = self.x_train[:, -i]
                TF_Genes = list(pd.Series(self.TF_Genes)[-i])
                config.input_shape = len(self.TF_Genes) - 1
            else:
                x_train = self.x_train
                TF_Genes = self.TF_Genes
            y_train = X_[:, target_index[i]].toarray()

            print("\nTarget gene: %s" % (target_Gene))
            # 2. Algorithm training
            t0 = time.time()
            config.train(x_train, y_train)
            t = (time.time() - t0) / 60

            print('Time spent: %.2f minutes' % t)

            Sel_Prob_Test, Sel_Binary_Test = config.Selected_Features(x_train)

            self.NN_predict(target_Gene, TF_Genes, Sel_Prob_Test, Sel_Binary_Test)
            i += 1


    def NN_predict(self, target_Gene, TF_Genes, Sel_Prob_Test, Sel_Binary_Test):
        ave_Prob = Sel_Prob_Test.mean(axis=0)
        acc_Binary = Sel_Binary_Test.sum(axis=0)
        for j, tf_name in enumerate(TF_Genes):
            self.TF2Gene_Binary = self.TF2Gene_Binary.append(
                {"TF": tf_name, "target": target_Gene, "vote": acc_Binary[j]}, ignore_index=True)
            self.TF2Gene_Prob = self.TF2Gene_Prob.append(
                {"TF": tf_name, "target": target_Gene, "probability": ave_Prob[j]}, ignore_index=True)


