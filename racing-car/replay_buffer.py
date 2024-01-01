import numpy as np

class ReplayBuffer():
    
    def __init__(self,max_size):
        # this will hold experiences
        self.storage = []
        # this is max size of storage
        self.max_size = max_size
        # indicator of current position of storage
        self.ptr = 0
        
    def push(self,data):
        #if len(self.storage) == self.max_size:
        #    self.storage[int(self.ptr)] = data
        #    self.ptr = (self.ptr + 1) % self.max_size
        #else:
        #    self.storage.append(data)
        if len(self.storage) == self.max_size:
            self.storage.pop(0)
            self.storage.append(data)
        else:
            self.storage.append(data)
    
    def sample(self, batch_size):
        return self.storage.pop(0)
        #state, next_state, action, reward, done = None, None, None, None, None

        #for i in enumerate(batch_size):
        #    st, n_st, act, rew, dn = self.storage.pop[0]
        #    state += st
        #    next_state += n_st
        #    action += act
        #    reward += rew
        #    done += dn

        #return np.array(state), np.array(next_state), np.array(action), np.array(reward).reshape(-1, 1), np.array(done).reshape(-1, 1)
