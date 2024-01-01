import gym
from ddpg import DDPG 
from exploration_noise import GaussianNoise

env = gym.make('CarRacing-v2',render_mode="human")
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
        next_state, reward, done, info,_ = env.step(action)
        env.render()
        if done: 
            print("reward{}".format(reward))
            print("Episode \t{}, the episode reward is \t{:0.2f}".format(i, ep_r))
        state = next_state