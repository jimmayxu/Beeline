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


"""Instance-wise Variable Selection (INVASE) module - with baseline

Reference: Jinsung Yoon, James Jordon, Mihaela van der Schaar, 
           "IINVASE: Instance-wise Variable Selection using Neural Networks," 
           International Conference on Learning Representations (ICLR), 2019.
Paper Link: https://openreview.net/forum?id=BJg_roAcK7
Contact: jsyoon0823@gmail.com
"""

# Necessary packages

class INVASE_config():
    """INVASE class.

    Attributes:
      - x_train: training features
      - y_train: training labels
      - model_type: invase or invase_minus
      - model_parameters:
        - actor_h_dim: hidden state dimensions for actor
        - critic_h_dim: hidden state dimensions for critic
        - n_layer: the number of layers
        - batch_size: the number of samples in mini batch
        - iteration: the number of iterations
        - activation: activation function of models
        - learning_rate: learning rate of model training
        - lamda: hyper-parameter of INVASE
    """
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
        self.lamda = 0.1
        self.actor_h_dim = 10
        self.critic_h_dim = 10
        self.n_layer = 3
        self.batch_size = batch_size
        self.iteration = iteration
        self.activation = activation
        self.learning_rate = 0.0001
        self.loss_type = loss_type

        self.dim = input_shape
        self.label_dim = output_shape

    def initialise(self):
        optimizer = Adam(self.learning_rate)
        # Build and compile critic
        self.critic = self.build_critic()
        self.critic.compile(loss=self.loss_type,
                            optimizer=optimizer, metrics=['acc'])

        # Build and compile the actor
        self.actor = self.build_actor()
        self.actor.compile(loss=self.actor_loss, optimizer=optimizer)

        # Build and compile the baseline
        self.baseline = self.build_baseline()
        self.baseline.compile(loss=self.loss_type,
                              optimizer=optimizer, metrics=['acc'])

    def actor_loss(self, y_true, y_pred):
        """Custom loss for the actor.

        Args:
          - y_true:
            - actor_out: actor output after sampling
            - critic_out: critic output
            - baseline_out: baseline output (only for invase)
          - y_pred: output of the actor network

        Returns:
          - loss: actor loss
        """

        # Actor output
        actor_out = y_true[:, :self.dim]
        # Critic output
        critic_out = y_true[:, self.dim:(self.dim + self.label_dim)]

        # Baseline output
        baseline_out = \
            y_true[:, (self.dim + self.label_dim):(self.dim + 2 * self.label_dim)]
        # Ground truth label
        y_out = y_true[:, (self.dim + 2 * self.label_dim):]


        if self.loss_type == 'poisson':
            critic_loss = -tf.reduce_sum(-critic_out + y_out * tf.math.log(critic_out + 1e-5), axis=1)
            baseline_loss = -tf.reduce_sum(-baseline_out + y_out * tf.math.log(baseline_out + 1e-5), axis=1)
        elif self.loss_type == 'categorical_crossentropy':
            critic_loss = -tf.reduce_sum(y_out * tf.math.log(critic_out + 1e-5), axis=1)
            baseline_loss = -tf.reduce_sum(y_out * tf.math.log(baseline_out + 1e-5), axis=1)
        elif self.loss_type == 'mean_squared_error':
            critic_loss = tf.reduce_mean(tf.math.square(critic_out - y_out))
            baseline_loss = tf.reduce_mean(tf.math.square(baseline_out - y_out))


        # Reward
        Reward = -(critic_loss - baseline_loss)

        # Policy gradient loss computation.
        custom_actor_loss = \
            Reward * tf.reduce_sum(actor_out * K.log(y_pred + 1e-5) + \
                                   (1 - actor_out) * K.log(1 - y_pred + 1e-5), axis=1) - \
            self.lamda * tf.reduce_mean(y_pred, axis=1)

        # custom actor loss
        custom_actor_loss = tf.reduce_mean(-custom_actor_loss)

        return custom_actor_loss

    def build_actor(self):
        """Build actor.

        Use feature as the input and output selection probability
        """
        actor_model = Sequential()
        actor_model.add(Dense(self.actor_h_dim, activation=self.activation,
                              kernel_regularizer=regularizers.l2(1e-3),
                              input_dim=self.dim))
        for _ in range(self.n_layer - 2):
            actor_model.add(Dense(self.actor_h_dim, activation=self.activation,
                                  kernel_regularizer=regularizers.l2(1e-3)))
        actor_model.add(Dense(self.dim, activation='sigmoid',
                              kernel_regularizer=regularizers.l2(1e-3)))

        feature = Input(shape=(self.dim,), dtype='float32')
        selection_probability = actor_model(feature)

        return Model(feature, selection_probability)

    def build_critic(self):
        """Build critic.

        Use selected feature as the input and predict labels
        """
        critic_model = Sequential()

        critic_model.add(Dense(self.critic_h_dim, activation=self.activation,
                               kernel_regularizer=regularizers.l2(1e-3),
                               input_dim=self.dim))
        critic_model.add(BatchNormalization())
        for _ in range(self.n_layer - 2):
            critic_model.add(Dense(self.critic_h_dim, activation=self.activation,
                                   kernel_regularizer=regularizers.l2(1e-3)))
            critic_model.add(BatchNormalization())
        critic_model.add(Dense(self.label_dim, activation='softmax',
                               kernel_regularizer=regularizers.l2(1e-3)))

        ## Inputs
        # Features
        feature = Input(shape=(self.dim,), dtype='float32')
        # Binary selection
        selection = Input(shape=(self.dim,), dtype='float32')

        # Element-wise multiplication
        critic_model_input = Multiply()([feature, selection])
        y_hat = critic_model(critic_model_input)

        return Model([feature, selection], y_hat)

    def build_baseline(self):
        """Build baseline.

        Use the feature as the input and predict labels
        """
        baseline_model = Sequential()

        baseline_model.add(Dense(self.critic_h_dim, activation=self.activation,
                                 kernel_regularizer=regularizers.l2(1e-3),
                                 input_dim=self.dim))
        baseline_model.add(BatchNormalization())
        for _ in range(self.n_layer - 2):
            baseline_model.add(Dense(self.critic_h_dim, activation=self.activation,
                                     kernel_regularizer=regularizers.l2(1e-3)))
            baseline_model.add(BatchNormalization())
        baseline_model.add(Dense(self.label_dim, activation='softmax',
                                 kernel_regularizer=regularizers.l2(1e-3)))

        # Input
        feature = Input(shape=(self.dim,), dtype='float32')
        # Output
        y_hat = baseline_model(feature)

        return Model(feature, y_hat)
    def Sample_M(self, gen_prob):

        # Shape of the selection probability
        n = gen_prob.shape[0]
        d = gen_prob.shape[1]

        # Sampling
        samples = np.random.binomial(1, gen_prob, (n, d))

        return samples
    def train(self, x_train, y_train):
        """Train INVASE.

        Args:
          - x_train: training features
          - y_train: training labels
        """
        self.initialise()
        for iter_idx in range(self.iteration):

            ## Train critic
            # Select a random batch of samples
            idx = np.random.randint(0, x_train.shape[0], self.batch_size)
            x_batch = x_train[idx, :]
            y_batch = y_train[idx, :]

            # Generate a batch of selection probability
            selection_probability = self.actor.predict(x_batch)
            # Sampling the features based on the selection_probability
            selection = self.Sample_M(selection_probability)
            # Critic loss
            critic_loss = self.critic.train_on_batch([x_batch, selection], y_batch)
            # Critic output
            critic_out = self.critic.predict([x_batch, selection.astype(x_batch.dtype)])


            # Baseline output
            # Baseline loss
            baseline_loss = self.baseline.train_on_batch(x_batch, y_batch)
            # Baseline output
            baseline_out = self.baseline.predict(x_batch)

            ## Train actor
            # Use multiple things as the y_true:
            # - selection, critic_out, baseline_out, and ground truth (y_batch)
            y_batch_final = np.concatenate((selection,
                                            np.asarray(critic_out),
                                            np.asarray(baseline_out),
                                            y_batch), axis=1)

            # Train the actor
            actor_loss = self.actor.train_on_batch(x_batch, y_batch_final)

            # Print the progress
            dialog = 'Iterations: ' + str(iter_idx) + \
                     ', critic loss: ' + str(critic_loss[0]) + \
                     ', baseline loss: ' + str(baseline_loss[0]) + \
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
