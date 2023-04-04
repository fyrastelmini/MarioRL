import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import cv2
import numpy as np
import os

from model import DQNAgent, create_dqn_model
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# Set up DQN agent
state_shape = (84, 84, 4)
action_shape = env.action_space.n
agent = DQNAgent()

# Prepopulate replay memory with saved frames
frames_dir = 'frames'
preprocess_shape = (84, 84)
preprocessed_frames = []
for frame_file in os.listdir(frames_dir):
    frame = cv2.imread(os.path.join(frames_dir, frame_file))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, preprocess_shape)
    preprocessed_frames.append(frame)
    if len(preprocessed_frames) == 4:
        state = np.stack(preprocessed_frames, axis=2)
        action = 0  # dummy action for demonstration
        reward = 0  # dummy reward for demonstration
        next_state = state  # next state is current state in this case
        done = False  # assume the game never ends
        agent.replay_memory.append((state, action, reward, next_state, done))
        preprocessed_frames.pop(0)  # remove oldest frame from buffer

# Train DQN agent
for episode in range(1000):
    state = env.reset()
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    state = cv2.resize(state, preprocess_shape)
    state = np.stack([state] * 4, axis=2)
    done = False
    total_reward = 0
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, info= env.step(action)
        next_state = cv2.cvtColor(next_state, cv2.COLOR_RGB2GRAY)
        next_state = cv2.resize(next_state, preprocess_shape)
        next_state = np.append(state[:,:,1:], np.expand_dims(next_state, axis=2), axis=2)
        agent.add_to_replay_memory(state, action, reward, next_state, done)
        agent.train()
        state = next_state
        total_reward += reward
    print(f'Episode {episode}, total reward: {total_reward}')