import gym
import gym_super_mario_bros
from model import create_dqn_model, DQNAgent
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = gym.wrappers.Monitor(env, './video', video_callable=lambda episode_id: True, force=True)
state_shape = env.observation_space.shape
action_shape = env.action_space.n
env.reset()

agent = DQNAgent()

num_episodes = 5000
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)
        agent.add_to_replay_memory(state, action, reward, next_state, done)
        agent.train()
        state = next_state
        total_reward += reward
    print(f'Episode: {episode+1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {agent.eps:.2f}')