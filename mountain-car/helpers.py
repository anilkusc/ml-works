import gymnasium as gym
from gymnasium.core import Env
import numpy as np
import random
import matplotlib.pyplot as plt

# this class stands for aggregating continous state to discrete states.
class StateAggregationEnv(gym.ObservationWrapper):
    #bins: piece number, low: lowest bound of states , high: highest bound of states
    def __init__(self, env: Env,bins,low,high):
        super().__init__(env)
        self.buckets = []
        # create pair for actions(location , velocity)
        pairs = zip(low, high, bins)
        for lower_limit, upper_limit, num_subdivisions in pairs:
            # linspace is partitioning the continious values to pieces.
            bucket = np.linspace(lower_limit, upper_limit, num_subdivisions - 1)
            self.buckets.append(bucket)
        # redefining observation space of gym environment  
        self.observation_space = gym.spaces.MultiDiscrete(nvec=bins.tolist())
    # make observed continious state to discrete.(overloading)
    # it will return qtable values for current observed value. [-1.1,0] -> (4,3).
    # [-1.1,0] is current observed location and velocity. (4,3) is indices of qtable. 4 is the row of qtalbe and 3 is the column
    def observation(self, state):
        indices = []
        for cont, buck in zip(state, self.buckets):
            index = np.digitize(cont, buck)
            indices.append(index)
        indices = tuple(indices)
        return indices

def initialize_game(position_discrete=20,velocity_discrete=20):

    env = gym.make('MountainCar-v0')
    bins = np.array([position_discrete,velocity_discrete])
    low = env.observation_space.low
    high = env.observation_space.high
    stateAggEnv = StateAggregationEnv(env,bins,low,high)
    q_table = np.zeros((position_discrete,velocity_discrete, stateAggEnv.action_space.n))

    return stateAggEnv,q_table

def find_action(epsilon,q_table,state,env):
     exp_tradeoff = random.uniform(0, 1)

     if exp_tradeoff > epsilon:
         # exploitation
         action = np.argmax(q_table[state])
     else:
         # exploration
         action = np.random.randint(env.action_space.n)
     return action

def qlearning(env,q_table,max_steps = 200,learning_rate = 0.1,discount_rate = 0.95,epsilon = 1,max_epsilon = 1,min_epsilon = 0.001,exploration_decay_rate = 0.005,num_episodes = 10000):
    total_rewards = []
    # for collecting data to visualize graph
    stats_control = num_episodes/100
    stats = 0
    episode_reward=0
    for episode in range(num_episodes):
        print("EPISODE " + str(episode) + "/"+str(num_episodes))
        state = env.reset()[0]
        done = False
        for step in range(max_steps):
            action = find_action(epsilon,q_table,state,env)
            new_state, reward, done, _, _ = env.step(action)
            q_table[state][action] = q_table[state][action] * (1 - learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[new_state]))
            state = new_state
            episode_reward += reward
            if done == True:
                break
        stats +=1
        if stats == stats_control:
            total_rewards.append(episode_reward/stats)
            stats = 0
            episode_reward=0
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-exploration_decay_rate*episode)
    episodes = range(0,int(num_episodes/stats_control))
    plt.plot(episodes, total_rewards, 'g', label='Rewards')
    plt.title('Reward Graph')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.legend()
    plt.show()
    return q_table

def play_game(env,q_table,num_episodes = 1000,max_steps = 200):
    env = gym.make('MountainCar-v0',render_mode="human")
    bins = np.array([20,20])
    low = env.observation_space.low
    high = env.observation_space.high
    env = StateAggregationEnv(env,bins,low,high)
    total_steps = []
    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        step=0
        print("****************************************************")
        print("EPISODE ", episode+1)
        for step in range(max_steps):
            action = np.argmax(q_table[state])
            new_state, _, done, _, _ = env.step(action)
            if done:
                print("You WIN")
                break
            state = new_state
        total_steps.append(step)
    # TO-DO : save graphic with matplotlib