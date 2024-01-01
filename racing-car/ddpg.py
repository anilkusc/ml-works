from replay_buffer import ReplayBuffer
from actor_critic_networks import Actor,Critic
import torch
import torch.nn.functional as F

class DDPG():
    
    def __init__(self, 
                 state_dim, 
                 action_dim,
                 capacity=1000000 , # will use in replay buffer.
                 tau=0.001, # will use for target network updating
                 batch_size=64, # will use to update target networks
                 update_iteration=200,
                 hidden_layer_actor=20,
                 hidden_layer_critic=64,
                 discount_factor=0.99
                 ):
        # parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.tau = tau
        self.batch_size = batch_size
        self.update_iteration = update_iteration
        self.hidden_layer_actor = hidden_layer_actor
        self.hidden_layer_critic = hidden_layer_critic 
        self.discount_factor = discount_factor
        
        # replay buffer
        self.replay_buffer = ReplayBuffer(capacity)

        # actor network
        self.actor = Actor(self.state_dim,self.action_dim,self.hidden_layer_actor)
        self.actor_target = Actor(self.state_dim,self.action_dim,self.hidden_layer_actor)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-3)

        # critic network
        self.critic = Critic(self.state_dim,self.action_dim,self.hidden_layer_critic)
        self.critic_target = Critic(self.state_dim,self.action_dim,self.hidden_layer_critic)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=2e-2)

    def select_action(self,state):
        state = torch.FloatTensor(state.reshape(1,-1))
        return self.actor(state)
    
    def update(self):
        for it in range(self.update_iteration):
            # For each Sample in replay buffer batch
            state, next_state, action, reward, done = self.replay_buffer.sample(self.batch_size)
            state = torch.FloatTensor(state.reshape(1,-1))
            action = torch.FloatTensor(action.reshape(1,-1))
            next_state = torch.FloatTensor(next_state.reshape(1,-1))
            done = torch.FloatTensor(1-done)
            reward = torch.FloatTensor([reward])
            # Compute the target Q value
            target_Q = self.critic_target(next_state, torch.FloatTensor(self.actor_target(next_state).reshape(1,-1)))
            target_Q = reward + (done * self.discount_factor * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Compute actor loss as the negative mean Q value using the critic network and the actor network
            actor_loss = -self.critic(state, torch.FloatTensor(self.actor(state).reshape(1,-1))).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            
            """
            Update the frozen target models using 
            soft updates, where 
            tau,a small fraction of the actor and critic network weights are transferred to their target counterparts. 
            """
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self):
        """
        Saves the state dictionaries of the actor and critic networks to files
        """
        torch.save(self.actor.state_dict(), 'actor.pth')
        torch.save(self.critic.state_dict(),'critic.pth')

    def load(self):
        """
        Loads the state dictionaries of the actor and critic networks to files
        """
        self.actor.load_state_dict(torch.load('actor.pth'))
        self.critic.load_state_dict(torch.load('critic.pth'))