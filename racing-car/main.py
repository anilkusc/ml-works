import gym
from ddpg import DDPG 
from exploration_noise import GaussianNoise

env = gym.make('CarRacing-v2')
env.reset()
env.render()

all_rewards = []
num_episodes = 100
num_trajectory = 5000
# getting input dimension for fitting to neural network
sd = env.observation_space.shape
state_dim = 1
for sd in env.observation_space.shape:
    state_dim = state_dim *sd


action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

agent = DDPG(state_dim, action_dim)
noiser = GaussianNoise(action_dim,max_action)


for i in range(num_episodes): 
    episode_total_reward = 0
    done = False
    state = env.reset()[0]
    trajectory = 0
    while not done:
        #action = env.action_space.sample()
        action = agent.select_action(state)
        #action = noiser.sample(action)
        next_state, reward, done, info,_ = env.step(action)
        if trajectory > num_trajectory:
            done = True
            agent.replay_buffer.push((state, next_state, action, -100, done))
        agent.replay_buffer.push((state, next_state, action, reward, done))
        state = next_state
        trajectory += 1

    all_rewards.append(episode_total_reward)
    agent.update()
    print("Training: " + str(i+1) + "/" + str(num_episodes))
agent.save()
env.close()

# test agent
env = gym.make('CarRacing-v2',render_mode="human")
env.reset()
env.render()
agent = DDPG(state_dim, action_dim)
agent.load()
for i in range(100):
    state = env.reset()[0]
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info,_ = env.step(action)
        env.render()
        state = next_state