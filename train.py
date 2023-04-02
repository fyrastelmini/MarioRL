import gym
from model import create_dqn_model, DQNAgent
env = gym.make('SuperMarioBros-v0')

state_shape = env.observation_space.shape
action_shape = env.action_space.n
