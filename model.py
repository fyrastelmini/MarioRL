import gym
import random
import numpy as np
import tensorflow as tf
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, RIGHT_ONLY)
state_shape = (84, 84, 4)
action_shape = env.action_space.n


def create_dqn_model():

    model = Sequential()
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=state_shape))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(action_shape, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=0.00025))
    return model

class DQNAgent:
    def __init__(self):
        self.model = create_dqn_model()
        self.target_model = create_dqn_model()
        self.replay_memory = []
        self.batch_size = 32
        self.gamma = 0.99
        self.eps = 1.0
        self.eps_decay = 0.9999
        self.eps_min = 0.01
        self.sync_interval = 10000
        self.steps = 0
        self.update_target_model()
        
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        
    def choose_action(self, state):
        if random.random() < self.eps:
            return env.action_space.sample()
        q_values = self.model.predict(np.expand_dims(state, axis=0))[0]
        return np.argmax(q_values)
        
    def add_to_replay_memory(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))
        
    def train(self):
        if len(self.replay_memory) < self.batch_size:
            return
        minibatch = random.sample(self.replay_memory, self.batch_size)
        states = np.zeros((self.batch_size, *state_shape))
        next_states = np.zeros((self.batch_size, *state_shape))
        actions, rewards, dones = [], [], []
        for i in range(self.batch_size):
            state, action, reward, next_state, done = minibatch[i]
            states[i] = state
            actions.append(action)
            rewards.append(reward)
            next_states[i] = next_state
            dones.append(done)
        q_values = self.model.predict(states)
        next_q_values = self.target_model.predict(next_states)
        for i in range(self.batch_size):
            if dones[i]:
                q_values[i][actions[i]] = rewards[i]
            else:
                q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        self.model.fit(states, q_values, batch_size=self.batch_size, verbose=0)
        self.steps += 1
        if self.steps % self.sync_interval == 0:
            self.update_target_model()
        self.eps = max(self.eps_min, self.eps * self.eps_decay)