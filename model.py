"""Actor/Critic Models for Deep Deterministic Policy Gradient."""

from keras import layers, models, optimizers, regularizers
from keras import backend as K
from collections import namedtuple, deque
import numpy as np
import random


class Actor:
    """Actor (Policy) model"""
    
    def __init__(self, state_size, action_size, action_low, action_high, learning_rate=1e-4):
        """Initialize parameters and build the model.

        Params
        ------
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """

        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low
        self.lr = learning_rate

        # Initialize other variables here

        # Build the model
        self.build_model()
    
    def build_model(self):
        """Build an actor (policy) model that maps states -> actions."""
        # Define input layer (state)
        states = layers.Input(shape=(self.state_size,), name='states')
        
        # Reshape action repeats into timesteps for recurrent layer
        reshape = layers.Reshape((9, 3))(states)

        # Add hidden layers
        net = layers.CuDNNLSTM(units=16, return_sequences=True)(reshape)
        net = layers.CuDNNLSTM(units=32)(net)
        net = layers.Dense(units=32, kernel_regularizer=regularizers.l2(0.01))(net)
        net = layers.BatchNormalization()(net)
        net = layers.LeakyReLU(alpha=0.1)(net)
        net = layers.Dense(units=64, kernel_regularizer=regularizers.l2(0.01))(net)
        net = layers.BatchNormalization()(net)
        net = layers.LeakyReLU(alpha=0.1)(net)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Add final output layer with sigmoid activation
        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid', name='raw_actions')(net)
        
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low, name='actions')(raw_actions)
        
        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = optimizers.Adam(lr=self.lr)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)


class Critic:
    """Critic (value) model."""

    def __init__(self, state_size, action_size, learning_rate=1e-5):
        """Initialize parameters and build model.

        Params
        ------
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate

        # Initialize any other variables here

        # Build the model
        self.build_model()
    
    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layers for state pathway
        reshape = layers.Reshape((9, 3))(states)
        net_states = layers.CuDNNLSTM(units=16)(reshape)
        net_states = layers.Dense(units=32)(states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.LeakyReLU(alpha=0.3)(net_states)
        net_states = layers.Dense(units=64)(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.LeakyReLU(alpha=0.3)(net_states)

        # Add hidden layers for action pathway
        net_actions = layers.Dense(units=32)(actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.LeakyReLU(alpha=0.3)(net_actions)
        net_actions = layers.Dense(units=64)(net_actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.LeakyReLU(alpha=0.3)(net_actions)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Add more layers to the combined network if needed
        net = layers.Dense(units=32)(net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation('relu')(net)

        # Add final output layer to produce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=self.lr)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an addition function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)


class ReplayBuffer:
    """Fixed-size circular buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object."""
        self.batch_size = batch_size  # size of training batch
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", 
                                                                "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=None, theta=0.15, sigma=0.3, dt=1e-2):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu * np.ones(self.size) if mu is not None else np.zeros(self.size)
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = self.mu

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn(len(x))
        self.state = x + dx
        return self.state
