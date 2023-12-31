class ReplayBuffer():
    
    def __init__(self,max_size):
        # this will hold experiences
        self.storage = []
        # this is max size of storage
        self.max_size = max_size
        # indicator of current position of storage
        self.ptr = 0
        
    def push(self,data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)
    
    def sample(self):
        pass
