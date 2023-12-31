import gym
from ddpg import DDPG 
from exploration_noise import GaussianNoise
import numpy as np

env = gym.make('CarRacing-v2',render_mode="human")
env.reset()
env.render()

all_rewards = []
num_episodes = 10000

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

agent = DDPG(state_dim, action_dim)
noiser = GaussianNoise(action_dim,max_action)


for _ in range(num_episodes): 
    episode_total_reward = 0
    done = False
    state = env.reset()[0]
    while not done:
        action = agent.select_action(state)
        action = noiser.sample(action)
        next_state, reward, done, info,_ = env.step(action)
        episode_total_reward += reward
        agent.replay_buffer.push((state, next_state, action, reward, done))
        state = next_state
    all_rewards.append(episode_total_reward)
env.close()
