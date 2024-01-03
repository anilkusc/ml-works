import gym
from ddpg import DDPG 
from exploration_noise import GaussianNoise
import random
import numpy as np
import os 

env = gym.make('MountainCarContinuous-v0')
all_rewards = []
num_episodes = 1000
num_trajectory = 1000
epsilon = 1
max_epsilon = 1
min_epsilon = 0.001
exploration_decay_rate = 0.0005
# getting input dimension for fitting to neural network
sd = env.observation_space.shape
state_dim = 1
for sd in env.observation_space.shape:
    state_dim = state_dim *sd

action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

agent = DDPG(state_dim, action_dim)
if os.path.exists("actor.pth"):
    agent.load()
noiser = GaussianNoise(action_dim,max_action)

for episode in range(num_episodes): 
    episode_total_reward = 0
    done = False
    state = env.reset()[0]
    trajectory = 0
    while not done:
        action = agent.select_action(state)
        # epsilon greedy
        if random.uniform(0, 1) < epsilon:
            action = noiser.sample(action)

        next_state, reward, done, info,_ = env.step(action)
        episode_total_reward += reward
        agent.replay_buffer.push((state, next_state, action, reward, float(done)))
        if trajectory > num_trajectory:
            done = True
        state = next_state
        trajectory += 1

    #all_rewards.append(episode_total_reward)
    agent.update()
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-exploration_decay_rate*episode)
    print("Training: " + str(episode+1) + "/" + str(num_episodes) + " Episode Total Reward: " + str(episode_total_reward) + ". Epsilon: " +str(epsilon) + " Step: " + str(trajectory))
agent.save()
env.close()

env = gym.make('MountainCarContinuous-v0',render_mode="human")
for i in range(100):
    state = env.reset()[0]
    done = False
    while not done:
        action = agent.select_action(state)
        
        next_state, reward, done, info,_ = env.step(action)
        print("action: " + str(action) + " Reward: " + str(reward))
        env.render()
        state = next_state