import gym
from ddpg import DDPG 
import numpy as np

env = gym.make('MountainCarContinuous-v0',render_mode="human")
state_dim = 1
for sd in env.observation_space.shape:
    state_dim = state_dim *sd

action_dim = env.action_space.shape[0]

env.reset()
env.render()
agent = DDPG(state_dim,action_dim)
agent.load()
for i in range(100):
    state = env.reset()[0]
    done = False
    while not done:
        action = agent.select_action(state)
        print(action)
        #action += -1.8
        next_state, reward, done, info,_ = env.step(np.float32(action))
        env.render()
        state = next_state