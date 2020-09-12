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
            iteration: int = 100, #20
            batch_size: int = 1000,
            loss_type: str = 'poisson',
            **kwargs
    ):
        self.TF_Genes = kwargs.get('TF_Genes', None)
        self.model = Feature_Selection_models(method_name='INVASE', **kwargs)

        self.config = INVASE_config(input_shape=len(self.TF_Genes), output_shape=1, activation='softplus',
                                    iteration=iteration, batch_size=batch_size, loss_type=loss_type)
    def fit(self,  X_):
        #with suppress_stdout():
        self.model.fit(X_=X_, config=self.config)
        return self.model.TF2Gene_Prob, self.model.TF2Gene_Binary

'''
Written by Jinsung Yoon
Date: Jan 1th 2019
INVASE: Instance-wise Variable Selection using Neural Networks Implementation on Synthetic Datasets
Reference: J. Yoon, J. Jordon, M. van der Schaar, "INVASE: Instance-wise Variable Selection using Neural Networks," International Conference on Learning Representations (ICLR), 2019.
Paper Link: https://openreview.net/forum?id=BJg_roAcK7
Contact: jsyoon0823@g.ucla.edu

------------------------------------- --------------

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
            iteration: int = 2000,
            batch_size: int = 1000,
            tau: float = 1.0,  # 1.5, #1.5 #0.01
            loss_type: str = 'poisson', # categorical_crossentropy, mean_squared_error
    ):

        self.latent_dim1 = 100      # Dimension of actor (selector) network
        self.latent_dim2 = 200      # Dimension of critic (discriminator) network

        self.batch_size = batch_size   # Batch size
        self.iteration = iteration         # Epoch size (large epoch is needed due to the policy gradient framework)
        self.lamda = 0.1            # Hyper-parameter for the number of selected features

        self.input_shape = input_shape     # Input dimension

        # final layer dimension
        self.output_shape = output_shape

        # Activation. (For Syn1 and 2, relu, others, selu)
        self.activation = activation

        self.loss_type = loss_type

    def initialise(self):
        # Use Adam optimizer with learning rate = 0.0001
        optimizer = Adam(0.0001)
        # Build and compile the discriminator (critic)
        self.critic = self.build_critic()
        # Use categorical cross entropy as the loss
        self.critic.compile(loss=self.loss_type, optimizer=optimizer, metrics=['acc'])

        # Build the selector (actor)
        self.actor = self.build_actor()
        # Use custom loss (my loss)
        self.actor.compile(loss=self.actor_loss, optimizer=optimizer)

        # Build and compile the value function
        self.baseline = self.build_baseline()
        # Use categorical cross entropy as the loss
        self.baseline.compile(loss=self.loss_type, optimizer=optimizer, metrics=['acc'])

    #%% Custom loss definition
    def actor_loss(self, y_true, y_pred):

        # dimension of the features
        d = y_pred.shape[1]

        # Put all three in y_true
        # 1. selected probability
        actor_out = y_true[:, :d]
        # 2. discriminator output
        critic_out = y_true[:, d:(d+self.output_shape)]
        # 3. valfunction output
        baseline_out = y_true[:, (d+self.output_shape):(d+self.output_shape*2)]
        # 4. ground truth
        y_out = y_true[:, (d+self.output_shape*2):]

        if self.loss_type == 'poisson':
            critic_loss = -tf.reduce_sum(-critic_out + y_out * tf.math.log(critic_out + 1e-8), axis=1)
            baseline_loss = -tf.reduce_sum(-baseline_out + y_out * tf.math.log(baseline_out + 1e-8), axis=1)
        elif self.loss_type == 'categorical_crossentropy':
            critic_loss = -tf.reduce_sum(y_out * tf.math.log(critic_out + 1e-8), axis=1)
            baseline_loss = -tf.reduce_sum(y_out * tf.math.log(baseline_out + 1e-8), axis=1)
        elif self.loss_type == 'mean_squared_error':
            critic_loss = tf.reduce_mean(tf.math.square(critic_out - y_out), axis=1)
            baseline_loss = tf.reduce_mean(tf.math.square(baseline_out - y_out), axis=1)

        Reward = -(critic_loss - baseline_loss)

        # B. Policy gradient loss computation.
        custom_actor_loss = Reward * tf.reduce_sum(actor_out * K.log(y_pred + 1e-8) + (1-actor_out) * K.log(1-y_pred + 1e-8), axis=1) - self.lamda * tf.reduce_mean(y_pred, axis=1)

        # C. custom actor loss
        custom_actor_loss = tf.reduce_mean(-custom_actor_loss)

        return custom_actor_loss

    #%% Generator (Actor)
    def build_actor(self):

        model = Sequential()

        model.add(Dense(100, activation=self.activation, name='s/dense1', kernel_regularizer=regularizers.l2(1e-3), input_dim=self.input_shape))
        model.add(Dense(100, activation=self.activation, name='s/dense2', kernel_regularizer=regularizers.l2(1e-3)))
        model.add(Dense(self.input_shape, activation='sigmoid', name='s/dense3', kernel_regularizer=regularizers.l2(1e-3)))

        model.summary()

        feature = Input(shape=(self.input_shape,), dtype='float32')
        select_prob = model(feature)

        return Model(feature, select_prob)

    #%% Discriminator (Critic)
    def build_critic(self):

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
    def build_baseline(self):

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
        self.initialise()
        # For each epoch (actually iterations)
        for iter_idx in range(self.iteration):

            #%% Train Discriminator
            # Select a random batch of samples
            idx = np.random.randint(0, x_train.shape[0], self.batch_size)
            x_batch = x_train[idx, :]
            y_batch = y_train[idx, :]

            # Generate a batch of probabilities of feature selection
            selection_probability = self.actor.predict(x_batch)

            # Sampling the features based on the generated probability
            selection = self.Sample_M(selection_probability)
            #selection = Sample_Concrete(self.tau, self.k)(gen_prob)

            # Compute the prediction of the critic based on the sampled features (used for selector training)
            critic_out = self.critic.predict([x_batch, selection.astype(x_batch.dtype)])

            # Train the discriminator
            critic_loss = self.critic.train_on_batch([x_batch, selection], y_batch)

            #%% Train Valud function

            # Compute the prediction of the critic based on the sampled features (used for selector training)
            baseline_out = self.baseline.predict(x_batch)

            # Train the discriminator
            baseline_loss = self.baseline.train_on_batch(x_batch, y_batch)

            #%% Train Generator
            # Use three things as the y_true: selection, critic_out, and ground truth (y_batch)
            y_batch_final = np.concatenate((selection, np.asarray(critic_out), np.asarray(baseline_out), y_batch), axis=1)

            # Train the selector
            actor_loss = self.actor.train_on_batch(x_batch, y_batch_final)


            #%% Plot the progress
            dialog = 'Iterations: ' + str(iter_idx) + \
                     ', critic accuracy: ' + str(critic_loss[1]) + \
                     ', baseline accuracy: ' + str(baseline_loss[1]) + \
                     ', actor loss: ' + str(np.round(actor_loss, 4))
            if iter_idx % 10 == 0:
                print(dialog)

    #%% Selected Features
    def Selected_Features(self, x_test):
        # determine the probability and binary selection output
        Sel_Prob_Test =  np.asarray(self.actor.predict(x_test))
        # Sel_Binary_Test = np.asarray(Sample_Concrete(self.tau, self.k).call_exp(Sel_Prob_Test))
        Sel_Binary_Test = np.asarray(self.Sample_M(Sel_Prob_Test))

        return Sel_Prob_Test, Sel_Binary_Test
