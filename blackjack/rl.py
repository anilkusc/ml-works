import numpy as np
import random
import json
import os

class RL:
    def __init__(self,
        win_reward = 1 ,
        draw_reward = 0 ,
        lost_reward = -1 ,
        continue_reward = 0 ,
        num_episodes = 100000 ,
        max_steps = 30 ,
        learning_rate = 0.7 ,
        discount_rate = 0.99 ,
        epsilon = 1 ,
        max_epsilon = 1 ,
        min_epsilon = 0.0001 ,
        exploration_decay_rate = 0.0001 ,
        ):

        if os.path.exists("q_table.json"):
            with open("q_table.json", "r") as json_file:
                q_table_json = json.load(json_file)
                self.q_table = np.array(q_table_json)
        else:
            self.q_table = self.create_states()
        print(self.q_table[0][0])
        self.win_reward = win_reward
        self.draw_reward = draw_reward
        self.lost_reward = lost_reward
        self.continue_reward = continue_reward
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.epsilon = epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.exploration_decay_rate = exploration_decay_rate

    def move(self, croupier_cards,player_cards):
        courpier_total = self.sum_of_cards(croupier_cards)
        player_total = self.sum_of_cards(player_cards)
            
        action = np.argmax(self.q_table[courpier_total][player_total])
        if action == 0:
            # 0 == stand
            #input("RL choose to stand. Please press any key")
            return False
        else:
            # 1 == hit
            #input("RL choose to hit. Please press any key")
            return True

    def train(self):
        for episode in range(self.num_episodes):
            # player's card can be max 21 after first delivery of 2 cards
            # courpier has only 1 open cards. It can be max 11
            state = [random.randint(1, 21),random.randint(1, 11)]
            # check if game is done
            done = False
            # is player standed once
            is_stand = False
            transitions = []
            global_reward = 0
            for step in range(self.max_steps):
                action,is_stand = self.find_action(state,is_stand)
                new_state, reward, done = self.step(state, action)
                transitions.append([state,action])
                state = new_state
                global_reward = reward + self.discount_rate * global_reward
                if done:
                    break
            average_reward = global_reward / len(transitions)
            for s,a in reversed(transitions):
                self.q_table[s[0],s[1], a] = self.q_table[s[0],s[1], a] * (1 - self.learning_rate) + self.learning_rate * average_reward
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.exploration_decay_rate*episode)
            completion_percentage = (episode + 1) / self.num_episodes * 100
            print(f"Episode {episode + 1}/{self.num_episodes} - Average Reward: {average_reward} - Completion: {completion_percentage:.2f}%")

        q_table_json = self.q_table.tolist()
        with open("q_table.json", "w") as json_file:
            json.dump(q_table_json, json_file)

    def find_possible_actions(self, state_array):
        possible_actions = []
        for i, s in enumerate(state_array):
            if s == 0:
                possible_actions.append(i)
        if len(possible_actions) == 0:
            possible_actions.append(-1)
        return possible_actions

    def create_states(self):
        rows = 30 # courpier's hand sum max
        cols = 30 # player's hand sum max
        actions = 2 # hit or stand
        table = np.zeros(shape=(rows, cols, actions))
        return table
    
    def step(self, current_state, action):
        # if player stands
        if action == 0:
            # courper's sum of card will be increased , player's cards will be same
            new_state = (random.randint(current_state[0], 29) ,current_state[1])
         # if player hits
        else:
            # player sum of card will be increased , courper's cards will be same
            new_state = (current_state[0],random.randint(current_state[1], 29),)
        reward, done = self.eval_state(current_state)
        return new_state, reward, done

    def eval_state(self, state):
        if state[1] == 21:
            return self.win_reward, True
        if state[0] == 21:
            return self.lost_reward, True
        if state[1] > 21:
            return self.lost_reward, True
        if state[0] > 21:
            return self.win_reward, True
        if 15 < state[0] < 22:
            if state[1] > state[0]:
                return self.win_reward, True
            else:
                return self.lost_reward, True
        else:
            return self.continue_reward,False

    def find_action(self,state,is_stand):
        exp_tradeoff = random.uniform(0, 1)
        
        if exp_tradeoff > self.epsilon:
            # exploitation
            action = np.argmax(self.q_table[state[0]][state[1]])

        else:
            # exploration
            action = random.randint(0, 1)
            # if player standed once set it true
            if action == 0:
                is_stand = True
        # if player standed once , all next states should be stand.
        if is_stand:
            action = 0        
        return action,is_stand

    def sum_of_cards(self, cards):
      total = 0
      has_ace = False

      for card in cards:
          if card != 1:
              total += card
          else:
              has_ace = True

      if has_ace:
          if total + 11 <= 21:
              total += 11
          else:
              total += 1

      return total