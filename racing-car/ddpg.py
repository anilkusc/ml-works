from replay_buffer import ReplayBuffer
from actor_critic_networks import Actor,Critic
import torch

class DDPG():
    
    def __init__(self, 
                 state_dim, 
                 action_dim,
                 capacity=1000000 , # will use in replay buffer.
                 batch_size=64,
                 update_iteration=200,
                 hidden_layer_actor=20,
                 hidden_layer_critic=64,
                 discount_factor=0.99
                 ):
        # parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.batch_size = batch_size
        self.update_iteration = update_iteration
        self.hidden_layer_actor = hidden_layer_actor
        self.hidden_layer_critic = hidden_layer_critic 
        self.discount_factor = discount_factor
        
        # replay buffer
        self.replay_buffer = ReplayBuffer(capacity)

        # actor network
        self.actor = Actor(self.state_dim,self.hidden_layer_actor)
        self.actor_target = Actor(self.state_dim,self.hidden_layer_actor)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-3)

        # critic network
        self.critic = Critic(self.state_dim,self.action_dim,self.hidden_layer_critic)
        self.critic_target = Critic(self.state_dim,self.action_dim,self.hidden_layer_critic)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=2e-2)

    def select_action(self,state):
        return 0
    
    def update(self):
        pass