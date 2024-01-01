import torch.nn as nn
import torch
import torch.nn.functional as F

# takes in a state observation as input and outputs an action, which is a continuous value.
class Actor(nn.Module):

    def __init__(self, environment_states, action_dim, hidden):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(environment_states, hidden), 
            nn.ReLU(), 
            nn.Linear(hidden, hidden), 
            nn.ReLU(), 
            nn.Linear(hidden, hidden), 
            nn.ReLU(), 
            nn.Linear(hidden, action_dim)
        )
        
    def forward(self, state):
        raw_output = self.net(state)
        
        # İlk çıkışı -1 ile 1 arasına sınırla (tanh kullan)
        output1 = torch.tanh(raw_output[:, 0])
        
        # İkinci çıkışı 0 ile 1 arasına sınırla (sigmoid kullan)
        output2 = F.sigmoid(raw_output[:, 1])
        
        # Üçüncü çıkışı 0 ile 1 arasına sınırla (sigmoid kullan)
        output3 = F.sigmoid(raw_output[:, 2])
        
        # Sonuçları birleştir
        bounded_outputs = torch.stack([output1, output2, output3], dim=1)
        
        # NumPy array'e dönüştür
        numpy_outputs = bounded_outputs.detach().cpu().numpy()
        
        return numpy_outputs[0]

#  takes in both a state observation and an action as input and outputs a Q-value, which estimates the expected total reward for the current state-action pair. 
class Critic(nn.Module):

    def __init__(self, environment_states, action_dim, hidden):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(environment_states + action_dim, hidden), 
            nn.ReLU(), 
            nn.Linear(hidden, hidden), 
            nn.ReLU(), 
            nn.Linear(hidden, hidden), 
            nn.ReLU(), 
            nn.Linear(hidden, 1)
        )
        
    def forward(self,state,action):
        sa = torch.cat((state, action),1)
        return self.net(sa)