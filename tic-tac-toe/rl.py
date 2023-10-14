import numpy as np
from itertools import product
import random

class RL:
    def __init__(self):
        self.type="rl"
        self.all_possible_states = self.create_states()
        self.action_space_size = 9
        self.state_space_size = len(self.all_possible_states)
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))
        self.game_board = [[0,0,0],[0,0,0],[0,0,0]]

        self.win_reward = 2
        self.draw_reward = 1
        self.lost_reward = 0
        self.num_episodes = 100
        self.max_steps = 9
        self.learning_rate = 0.5
        self.discount_rate = 0.99
        self.epsilon = 1
        self.max_epsilon = 1
        self.min_epsilon = 0.0001
        self.exploration_decay_rate = 0.0001

    def train(self):
        for episode in range(self.num_episodes):
            state = 0
            done = False
            for step in range(self.max_steps):
                exp_tradeoff = random.uniform(0, 1)
                possible_actions = self.find_possible_actions()
                if exp_tradeoff > self.epsilon:
                    # exploitation
                    action = np.argmax(self.q_table[state, :])
                else:
                    # exploration                    
                    action = random.choice(possible_actions)
                new_state, reward, done = self.step(state,action)
                self.q_table[state, action] = self.q_table[state, action] * (1 - self.learning_rate) + self.learning_rate * (reward + self.discount_rate * np.max(self.q_table[new_state, :]))
                state = new_state
                if done == True:
                    break
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.exploration_decay_rate*episode)
        print(self.q_table)

    
    def find_possible_actions(self):
        possible_actions = []
        state_array = self.convert_state_to_array(self.game_board)
        for i,s in enumerate(state_array):
            if s == 0:
                possible_actions.append(i)
        if len(possible_actions) == 0:
            possible_actions.append[-1]
        return possible_actions

    def create_states(self):
        rows = 3
        cols = 3
        
        values = [0, 1, 2]
        all_states_3x3 = []
        all_states = list(product(values, repeat=rows * cols))
        for state in all_states:
            all_states_3x3.append([[state[0],state[1],state[2]],[state[3],state[4],state[5]],[state[6],state[7],state[8]]])

        return all_states_3x3

    def eval_state(self,state):
        if self.is_winner(state):
            return self.win_reward,True
        if self.is_draw(state):
            return self.draw_reward,True
        return self.lost_reward,False

    def step(self,current_state,action):
        current_table = self.all_possible_states[current_state]
        next_state = self.convert_state_to_array(current_table)
        next_state[action] = 1
        for i,state in enumerate(self.all_possible_states):
            if self.convert_state_to_array(state) == next_state:
                new_state = i
                new_table = state
        reward, done = self.eval_state(self.convert_state_to_array(new_table))
        return new_state, reward, done

    def convert_state_to_array(self,state):
        convert_to_array = []
        for row in state:
            for item in row:
                convert_to_array.append(item)
        return convert_to_array

    def convert_array_to_state(self,array):
        return [[array[0],array[1],array[2]],[array[3],array[4],array[5]],[array[6],array[7],array[8]]]
    
    def is_winner(self,action):
        board = self.convert_array_to_state(action)
        for row in board:
            if row[0] == row[1] == row[2] == 1:
                return True
        for i,_ in enumerate(self.game_board):
            if self.game_board[0][i] == self.game_board[1][i] == self.game_board[2][i] == 1:
                return True
            if self.game_board[0][0] == self.game_board[1][1] == self.game_board[2][2] == 1:
                return True
            if self.game_board[0][2] == self.game_board[1][1] == self.game_board[2][0] == 1:
                return True
        return False        

    def is_draw(self,action):
      board = self.convert_array_to_state(action)
      for row in board:
        for item in row:
          if item == 0:
            return False
      return True    

rl = RL()
rl.train()
print(rl)