"""
INVASE (Instance-wise feature selection method)
"""
#%% Necessary packages
# 1. Keras
from tensorflow.keras.layers import Input, Dense, Multiply
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

# 2. Others
import tensorflow as tf
import numpy as np
import pandas as pd
import sys, os


# 3. Benchmarking specific
from Algorithms.INVASE import Feature_Selection_models
from contextlib import contextmanager
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


#%% Define INVASE class
class INVASE:
    def __init__(
            self,
            folder_name: str = 'toy',
            epochs: int = 100, #20
            batch_size: int = 1000,
            **kwargs
    ):
        self.TF_Genes = kwargs.get('TF_Genes', None)
        self.model = Feature_Selection_models(method_name='INVASE', folder_name=folder_name, **kwargs)
        with suppress_stdout():
            self.config = INVASE_config(input_shape=len(self.TF_Genes), output_shape=1, activation='softplus',
                           epochs=epochs, batch_size=batch_size)
    def fit(self,  X_):
        doTrainTest = False
        TF2gene = self.model.fit(X_=X_, config=self.config, doTrainTest=doTrainTest)
        return TF2gene


'''
Written by Jinsung Yoon
Date: Jan 1th 2019
INVASE: Instance-wise Variable Selection using Neural Networks Implementation on Synthetic Datasets
Reference: J. Yoon, J. Jordon, M. van der Schaar, "INVASE: Instance-wise Variable Selection using Neural Networks," International Conference on Learning Representations (ICLR), 2019.
Paper Link: https://openreview.net/forum?id=BJg_roAcK7
Contact: jsyoon0823@g.ucla.edu

---------------------------------------------------

Instance-wise Variable Selection (INVASE) - with baseline networks
'''


#%% Define INVASE configure class
class INVASE_config():

    # 1. Initialization

    def __init__(
            self,
            input_shape,
            output_shape=1, #2
            activation='softplus', #'selu'
            epochs: int = 2000,
            batch_size: int = 1000,
            tau: float = 1.0,  # 1.5, #1.5 #0.01
    ):

        self.tau = tau

        self.latent_dim1 = 100      # Dimension of actor (selector) network
        self.latent_dim2 = 200      # Dimension of critic (discriminator) network

        self.batch_size = batch_size   # Batch size
        self.epochs = epochs         # Epoch size (large epoch is needed due to the policy gradient framework)
        self.lamda = 0.1            # Hyper-parameter for the number of selected features

        self.input_shape = input_shape     # Input dimension

        # final layer dimension
        self.output_shape = output_shape

        # Activation. (For Syn1 and 2, relu, others, selu)
        self.activation = activation

        # Use Adam optimizer with learning rate = 0.0001
        optimizer = Adam(0.0001)

        # Build and compile the discriminator (critic)
        self.discriminator = self.build_discriminator()
        # Use categorical cross entropy as the loss
        # self.discriminator.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
        self.discriminator.compile(loss='poisson', optimizer=optimizer, metrics=['acc'])
        # self.discriminator.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['acc'])

        # Build the selector (actor)
        self.selector = self.build_selector()
        # Use custom loss (my loss)
        self.selector.compile(loss=self.my_loss, optimizer=optimizer)

        # Build and compile the value function
        self.valfunction = self.build_valfunction()
        # Use categorical cross entropy as the loss
        # self.valfunction.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
        self.valfunction.compile(loss='poisson', optimizer=optimizer, metrics=['acc'])
        # self.valfunction.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['acc'])

    #%% Custom loss definition
    def my_loss(self, y_true, y_pred):

        # dimension of the features
        d = y_pred.shape[1]

        # Put all three in y_true
        # 1. selected probability
        sel_prob = y_true[:, :d]
        # 2. discriminator output
        dis_prob = y_true[:, d:(d+self.output_shape)]
        # 3. valfunction output
        val_prob = y_true[:, (d+self.output_shape):(d+self.output_shape*2)]
        # 4. ground truth
        y_final = y_true[:, (d+self.output_shape*2):]

        # A1. Compute the rewards of the actor network
        # change the reward to poisson log likelihood function
        # Reward1 = tf.reduce_sum(y_final * tf.math.log(dis_prob + 1e-8), axis=1)
        Reward1 = tf.reduce_sum(- dis_prob + y_final * tf.math.log(dis_prob + 1e-4), axis=1)
        # A2. Compute the rewards of the actor network
        # Reward2 = tf.reduce_sum(y_final * tf.math.log(val_prob + 1e-8), axis=1)
        Reward2 = tf.reduce_sum(- val_prob + y_final * tf.math.log(val_prob + 1e-4), axis=1)
        # Difference is the rewards
        Reward = Reward1 - Reward2
        Reward = tf.cast(Reward, tf.float32)

        # B. Policy gradient loss computation.
        loss1 = Reward * tf.reduce_sum(sel_prob * K.log(y_pred + 1e-4) + (1-sel_prob) * K.log(1-y_pred + 1e-4), axis=1) - self.lamda * tf.reduce_mean(y_pred, axis=1)

        # C. Maximize the loss1
        loss = tf.reduce_mean(-loss1)

        return loss

    #%% Generator (Actor)
    def build_selector(self):

        model = Sequential()

        model.add(Dense(100, activation=self.activation, name='s/dense1', kernel_regularizer=regularizers.l2(1e-3), input_dim=self.input_shape))
        model.add(Dense(100, activation=self.activation, name='s/dense2', kernel_regularizer=regularizers.l2(1e-3)))
        model.add(Dense(self.input_shape, activation='sigmoid', name='s/dense3', kernel_regularizer=regularizers.l2(1e-3)))

        model.summary()

        feature = Input(shape=(self.input_shape,), dtype='float32')
        select_prob = model(feature)

        return Model(feature, select_prob)

    #%% Discriminator (Critic)
    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(200, activation=self.activation, name='dense1', kernel_regularizer=regularizers.l2(1e-3), input_dim=self.input_shape))
        model.add(BatchNormalization())     # Use Batch norm for preventing overfitting
        model.add(Dense(200, activation=self.activation, name='dense2', kernel_regularizer=regularizers.l2(1e-3)))
        model.add(BatchNormalization())
        model.add(Dense(self.output_shape, activation='softmax', name='dense3', kernel_regularizer=regularizers.l2(1e-3)))

        model.summary()

        # There are two inputs to be used in the discriminator
        # 1. Features
        feature = Input(shape=(self.input_shape,), dtype='float32')
        # 2. Selected Features
        select = Input(shape=(self.input_shape,), dtype='float32')

        # Element-wise multiplication
        model_input = Multiply()([feature, select])
        prob = model(model_input)

        return Model([feature, select], prob)

    #%% Value Function
    def build_valfunction(self):

        model = Sequential()

        model.add(Dense(200, activation=self.activation, name='v/dense1', kernel_regularizer=regularizers.l2(1e-3), input_dim=self.input_shape))
        model.add(BatchNormalization())     # Use Batch norm for preventing overfitting
        model.add(Dense(200, activation=self.activation, name='v/dense2', kernel_regularizer=regularizers.l2(1e-3)))
        model.add(BatchNormalization())
        model.add(Dense(self.output_shape, activation='softmax', name='v/dense3', kernel_regularizer=regularizers.l2(1e-3)))


        model.summary()

        # There are one inputs to be used in the value function
        # 1. Features
        feature = Input(shape=(self.input_shape,), dtype='float32')

        # Element-wise multiplication
        prob = model(feature)

        return Model(feature, prob)

    #%% Sampling the features based on the output of the selector
    def Sample_M(self, gen_prob):

        # Shape of the selection probability
        n = gen_prob.shape[0]
        d = gen_prob.shape[1]

        # Sampling
        samples = np.random.binomial(1, gen_prob, (n, d))

        return samples

    #%% Training procedure
    def train(self, x_train, y_train, x_val=None, y_val=None):

        y_train = y_train.astype('float32')
        loss = pd.DataFrame()
        # For each epoch (actually iterations)
        for epoch in range(self.epochs):

            #%% Train Discriminator
            # Select a random batch of samples
            idx = np.random.randint(0, x_train.shape[0], self.batch_size)
            x_batch = x_train[idx, :]
            y_batch = y_train[idx, :]

            # Generate a batch of probabilities of feature selection
            gen_prob = self.selector.predict(x_batch)

            # Sampling the features based on the generated probability
            sel_prob = self.Sample_M(gen_prob)

            # Compute the prediction of the critic based on the sampled features (used for selector training)
            dis_prob = self.discriminator.predict([x_batch, sel_prob])

            # Train the discriminator
            d_loss = self.discriminator.train_on_batch([x_batch, sel_prob], y_batch)

            #%% Train Valud function

            # Compute the prediction of the critic based on the sampled features (used for selector training)
            val_prob = self.valfunction.predict(x_batch)

            # Train the discriminator
            v_loss = self.valfunction.train_on_batch(x_batch, y_batch)

            #%% Train Generator
            # Use three things as the y_true: sel_prob, dis_prob, and ground truth (y_batch)
            y_batch_final = np.concatenate((sel_prob, np.asarray(dis_prob), np.asarray(val_prob), y_batch), axis=1)

            # Train the selector
            g_loss = self.selector.train_on_batch(x_batch, y_batch_final)


            #%% Plot the progress
            dialog = 'Epoch: ' + str(epoch) + ', d_loss (Acc)): ' + str(d_loss[1]) + ', v_loss (Acc): ' + str(v_loss[1]) + ', g_loss: ' + str(np.round(g_loss, 4))

            if epoch % 10 == 0:
                print(dialog)
                loss = loss.append({'d_loss': d_loss[1],
                                    'v_loss': v_loss[1],
                                    'g_loss': g_loss},
                                   ignore_index=True)
        return loss

    #%% Selected Features
    def Selected_Features(self, x_test):
        # determine the probability and binary selection output
        Sel_Prob_Test = np.asarray(self.selector.predict(x_test))
        Sel_Binary_Test = np.asarray(self.Sample_M(Sel_Prob_Test))

        return Sel_Prob_Test, Sel_Binary_Test

