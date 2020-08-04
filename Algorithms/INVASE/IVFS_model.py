import numpy as np
import pandas as pd
import os, time
from tqdm import tqdm
import psutil
from scipy.special import softmax


class Feature_Selection_models:
    def __init__(
            self,
            method_name,
            folder_name: str = 'toy', 
            rerun: bool = False,
            **kwargs
    ):
        self.method_name = method_name
        self.rerun = rerun

        self.i = kwargs.get('i', '')
        self.CellLabel = kwargs.get('CellLabel', None)

        self.target_Genes = kwargs.get('target_Genes', None)
        self.TF_Genes = kwargs.get('TF_Genes', None)
        self.gene_names = kwargs.get('gene_names', None)
        self.save_path = os.path.join('saved_model', method_name, folder_name)
        assert self.TF_Genes is not None
        assert all(pd.Series(self.TF_Genes).isin(self.gene_names))

        os.makedirs(self.save_path, exist_ok=True)
        self.saved_path = os.path.join(self.save_path, str(self.i))
        os.makedirs(self.saved_path, exist_ok=True)

        self.kwargs = kwargs

        self.save_file_Prob = os.path.join(self.saved_path, "TF2Gene_Prob.csv")
        self.save_file_Binary = os.path.join(self.saved_path, "TF2Gene_Binary.csv")


    def fit(self, X_, config, doTrainTest=False):
        assert pd.Series(self.target_Genes).isin(self.gene_names).all()
        ## TRACK MEMORY FUNCTION #####
        open(os.path.join(self.saved_path, "Memory.tsv"), "w").write("Step\tMemory_MB\tTime\n")
        self.logMem("Start method implementation")
        index = np.where(self.gene_names.isin(self.TF_Genes))[0]
        target_index = np.where(self.gene_names.isin(self.target_Genes))[0]

        if doTrainTest:
            train_idx, test_idx, val_idx = self.TrainTestSet(testSet=0.2)
            x_val = X_[np.ix_(val_idx, index)].toarray()
            y_temp = X_[train_idx, target_index[0]].toarray()
            x_train = X_[np.ix_(train_idx, index)].toarray()
            self.x_test = X_[np.ix_(test_idx, index)].toarray()
            self.test_idx = test_idx
        else:
            x_train = X_[:, index].toarray()
            x_val = None
            #x_val = x_train.reshape(-1, x_train.shape[1], 1) if self.method_name is 'ASAC' else x_train.copy()
            y_temp = X_[:, target_index[0]].toarray()
            self.x_test = x_train.copy()
            self.test_idx = None

        y_scale = softmax(y_temp).reshape(-1, 1)
        y_train = np.concatenate((y_scale, 1 - y_scale), 1)
        #y_train = y_temp #!!!!!!

        if os.path.isfile(self.save_file_Prob) and os.path.isfile(self.save_file_Binary) and not self.rerun:
            print('\nLoad the result for %s' %self.method_name)
            self.TF2Gene_Prob = pd.read_csv(self.save_file_Prob)
            self.TF2Gene_Binary = pd.read_csv(self.save_file_Binary)
        else:
            print('\nTraining for %s' % self.method_name)
            self.bin_Prob = dict()
            self.bin_Binary = dict()
            self.TF2Gene_Prob = pd.DataFrame(columns=["TF", "target", "probability"])
            self.TF2Gene_Binary = pd.DataFrame(columns=["TF", "target", "vote"])
            """
            # Pre-training
            print("Pre-training model")
            t = time.time()
            for i in tqdm(
                    range(10),
                    desc=("Pre-training %s model" % self.method_name)
            ):
                config.train(x_train, y_train)

            info = 'Time_%.3fspe(%d)' % ((time.time() - t)/config.epochs/10, config.epochs)
            os.makedirs(os.path.join(self.saved_path, info), exist_ok=True)
            mode = 'Training'
            self.logMem("Pre-Trained")
            """

            i = 0
            for target_Gene in tqdm(
                    self.target_Genes,
                    desc=("Training %s model for each target gene" % (self.method_name))
            ):
                if doTrainTest:
                    y_temp = X_[train_idx, target_index[i]].toarray()
                    y_temp_val = X_[val_idx, target_index[i]].toarray()
                    y_scale = softmax(y_temp_val).reshape(-1, 1)
                    y_val = np.concatenate((y_scale, 1 - y_scale), 1)
                    #y_val = y_temp_val #!!!!!!

                else:
                    y_temp = X_[:, target_index[i]].toarray()
                    y_val = None
                """
                y_scale = softmax(y_temp).reshape(-1, 1)
                y_train = np.concatenate((y_scale, 1 - y_scale), 1)
                """
                y_train = y_temp

                print("\nTarget gene: %s" % (target_Gene))
                # 2. Algorithm training
                self.logMem("--Model Setup")
                t0 = time.time()
                config.train(x_train, y_train, x_val, y_val)
                t = (time.time() - t0) / 60
                self.logMem("--config Trained for %s" %target_Gene)

                print('Time spent: %.2f minutes' % t)

                Sel_Prob_Test, Sel_Binary_Test = config.Selected_Features(self.x_test)

                self.NN_predict(target_Gene, Sel_Prob_Test, Sel_Binary_Test)
                i += 1

            self.logMem("Models are trained for all target genes")

            super().__init__(
                gene_names=self.gene_names
            )

            self.logMem("Selection probability matrix and selection binary matrix in cell bins is saved")
            self.TF2Gene_Prob.to_csv(self.save_file_Prob, index=False)
            self.TF2Gene_Binary.to_csv(self.save_file_Binary, index=False)

        return self.TF2Gene_Prob

    def NN_predict(self, target_Gene, Sel_Prob_Test, Sel_Binary_Test):
        ave_Prob = Sel_Prob_Test.mean(axis=0)
        acc_Binary = Sel_Binary_Test.sum(axis=0)
        for j, tf_name in enumerate(self.TF_Genes):
            self.TF2Gene_Binary = self.TF2Gene_Binary.append(
                {"TF": tf_name, "target": target_Gene, "vote": acc_Binary[j]}, ignore_index=True)
            self.TF2Gene_Prob = self.TF2Gene_Prob.append(
                {"TF": tf_name, "target": target_Gene, "probability": ave_Prob[j]}, ignore_index=True)


    def logMem(self, text):
        open(os.path.join(self.saved_path, "Memory.tsv"), "a").write(
            text + "\t" + str(psutil.Process(os.getpid()).memory_info().rss * 1.0 / 10 ** 6) + "\t" + time.strftime(
                "%Y-%m-%d %H:%M:%S", time.gmtime()) + "\n")
