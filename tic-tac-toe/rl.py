import numpy as np
from itertools import product
import random


class RL:
    def __init__(self):
        self.type = "rl"
        self.all_possible_states = self.create_states()
        self.action_space_size = 9
        self.state_space_size = len(self.all_possible_states)
        self.q_table = np.zeros(
            (self.state_space_size, self.action_space_size))

        self.win_reward = 100
        self.draw_reward = 10
        self.lost_reward = 0
        self.continue_reward = 1
        self.num_episodes = 10000
        self.max_steps = 9
        self.learning_rate = 0.7
        self.discount_rate = 0.99
        self.epsilon = 1
        self.max_epsilon = 1
        self.min_epsilon = 0.0001
        self.exploration_decay_rate = 0.0001

    def move(self, current_table):
        for i, t in enumerate(self.all_possible_states):
            if t == current_table:
                state = i
                break
        current_state = self.convert_state_to_array(current_table)
        possible_actions = []
        for i, q_value in enumerate(self.q_table[state, :]):
            if current_state[i] == 0:
                possible_actions.append(q_value)
            else:
                possible_actions.append(-1)
        action = np.argmax(possible_actions)
        if action == 0:
            return 0, 0
        elif action == 1:
            return 0, 1
        elif action == 2:
            return 0, 2
        elif action == 3:
            return 1, 0
        elif action == 4:
            return 1, 1
        elif action == 5:
            return 1, 2
        elif action == 6:
            return 2, 0
        elif action == 7:
            return 2, 1
        elif action == 8:
            return 2, 2

    def train(self):
        for episode in range(self.num_episodes):
            print("%", episode*100/self.num_episodes)
            state = 0
            done = False
            for step in range(self.max_steps):
                exp_tradeoff = random.uniform(0, 1)
                current_state = self.convert_state_to_array(self.all_possible_states[state])
                possible_actions = self.find_possible_actions(current_state)
                
                if exp_tradeoff > self.epsilon:
                    # exploitation                   
                    possible_actions = []
                    for i, q_value in enumerate(self.q_table[state, :]):
                        if current_state[i] == 0:
                            possible_actions.append(q_value)
                        else:
                            possible_actions.append(-1)
                    action = np.argmax(possible_actions)
                else:
                    # exploration
                    action = random.choice(possible_actions)
                new_state, reward, done = self.step(state, action)
                self.q_table[state, action] = self.q_table[state, action] * (1 - self.learning_rate) + self.learning_rate * (
                    reward + self.discount_rate * np.max(self.q_table[new_state, :]))
                state = new_state
                if done == True:
                    break
            self.epsilon = self.min_epsilon + \
                (self.max_epsilon - self.min_epsilon) * \
                np.exp(-self.exploration_decay_rate*episode)

    def find_possible_actions(self, state_array):
        possible_actions = []
        for i, s in enumerate(state_array):
            if s == 0:
                possible_actions.append(i)
        if len(possible_actions) == 0:
            possible_actions.append(-1)
        return possible_actions

    def create_states(self):
        rows = 3
        cols = 3
        
        values = [0, 1, 2]
        all_states_3x3 = []
        all_states = list(product(values, repeat=rows * cols))
        for state in all_states:
            #zero_indices = [i for i, value in enumerate(state) if value == 0]
            one_indices = state.count(1)
            two_indices = state.count(2)
            if one_indices == two_indices or one_indices == two_indices + 1 or two_indices == one_indices+1:
                all_states_3x3.append([[state[0],state[1],state[2]],[state[3],state[4],state[5]],[state[6],state[7],state[8]]])

        return all_states_3x3

    def eval_state(self, state):
        if self.is_winner(state):
            return self.win_reward, True
        if self.is_draw(state):
            return self.draw_reward, True
        if self.is_lost(state):
            return self.lost_reward, True
        return self.continue_reward, False

    def step(self, current_state, action):
        current_table = self.all_possible_states[current_state]
        next_state = self.convert_state_to_array(current_table)
        next_state[action] = 1
        # Player 2 plays randomly
        possible_actions = self.find_possible_actions(next_state)
        if len(possible_actions) == 1 and possible_actions[0] == -1:
            pass
        else:
            next_state[random.choice(possible_actions)] = 2
        for i, state in enumerate(self.all_possible_states):
            if self.convert_state_to_array(state) == next_state:
                new_state = i
                new_table = state
                break
        reward, done = self.eval_state(self.convert_state_to_array(new_table))
        return new_state, reward, done

    def convert_state_to_array(self, state):
        convert_to_array = []
        for row in state:
            for item in row:
                convert_to_array.append(item)
        return convert_to_array

    def convert_array_to_state(self, array):
        return [[array[0], array[1], array[2]], [array[3], array[4], array[5]], [array[6], array[7], array[8]]]

    def is_winner(self, state):
        if state[0] == state[1] == state[2] == 1 or state[3] == state[4] == state[5] == 1 or state[6] == state[7] == state[8] == 1:
            return True
        if state[0] == state[3] == state[6] == 1 or state[1] == state[4] == state[7] == 1 or state[2] == state[5] == state[8] == 1:
            return True
        if state[0] == state[3] == state[6] == 1 or state[1] == state[4] == state[7] == 1 or state[2] == state[5] == state[8] == 1:
            return True
        if state[0] == state[4] == state[8] == 1 or state[2] == state[4] == state[6] == 1:
            return True
        return False

    def is_lost(self, state):
        if state[0] == state[1] == state[2] == 2 or state[3] == state[4] == state[5] == 2 or state[6] == state[7] == state[8] == 2:
            return True
        if state[0] == state[3] == state[6] == 2 or state[1] == state[4] == state[7] == 2 or state[2] == state[5] == state[8] == 2:
            return True
        if state[0] == state[3] == state[6] == 2 or state[1] == state[4] == state[7] == 2 or state[2] == state[5] == state[8] == 2:
            return True
        if state[0] == state[4] == state[8] == 2 or state[2] == state[4] == state[6] == 2:
            return True
        return False

    def is_draw(self, state):
        for item in state:
            if item == 0:
                return False
        return True
