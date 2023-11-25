import random

class Game:
  def __init__(self,player):

    self.cards = self.shuffle_cards()
    self.croupier_cards = []
    self.player_cards = []
    self.is_turn_over = False
    self.player = player
    self.ai_wins = 0
    self.courpier_wins = 0
    self.total_game_count = 1000

  def shuffle_cards(self):
    card_values = [
      1,1,1,1,  # Ace
      2,2,2,2, 
      3,3,3,3, 
      4,4,4,4, 
      5,5,5,5, 
      6,6,6,6, 
      7,7,7,7, 
      8,8,8,8, 
      9,9,9,9, 
      10,10,10,10,  # 10 
      10,10,10,10,  # Jack
      10,10,10,10,  # Queen
      10,10,10,10   # King 
    ]
    random.shuffle(card_values)
    return card_values

  def start(self):
    i = 0
    while True:
      i += 1
      print("#############################################")
      print("Starting new turn")
      self.turn()
      self.is_turn_over = False
      if len(self.cards) < 10:
        print("#############################################")
        print("Reshufling cards")
        self.cards = self.shuffle_cards()
      if i > self.total_game_count:
         print("Wins:",self.ai_wins)
         print("Lost:",self.courpier_wins)
         break


  def turn(self):
    self.deliver_first_time()
    self.print_current_status_censored()
    
    while not self.is_turn_over and self.player.is_hit(self.croupier_cards,self.player_cards):
      self.player_cards.append(self.cards.pop(0))    
      self.evaluate_players_card()
      self.print_current_status_censored()

    while not self.is_turn_over:
        self.print_current_status()
        self.evaluate_courpiers_card()
        if not self.is_turn_over:
          print("popping new card!")
          self.croupier_cards.append(self.cards.pop(0))
          print("#############################################")
          
    self.evaluate_game()
    print("#############################################")
    self.print_current_status()
    print("#############################################")
    self.croupier_cards = []
    self.player_cards = []


  def deliver_first_time(self):
    self.player_cards.append(self.cards.pop(0))
    self.croupier_cards.append(self.cards.pop(0))
    self.player_cards.append(self.cards.pop(0))
    self.croupier_cards.append(self.cards.pop(0))

  def print_current_status_censored(self):
    print("---------croupier's hand---------")
    card_vis = "#"
    for i,card in enumerate(self.croupier_cards):
      if len(self.croupier_cards)-1 == i:
        card_vis = card_vis +" ? #"
      else:
        card_vis = card_vis +" "+str(card)+" #"
    print(card_vis)
    card_vis = "#"
    print("---------player's hand---------")
    for card in self.player_cards:
        card_vis = card_vis +" "+str(card)+" #"
    print(card_vis)

  def print_current_status(self):
    print("---------croupier's hand---------")
    card_vis = "#"
    for i,card in enumerate(self.croupier_cards):
        card_vis = card_vis +" "+str(card)+" #"
    print(card_vis)
    card_vis = "#"
    print("---------player's hand---------")
    for card in self.player_cards:
        card_vis = card_vis +" "+str(card)+" #"
    print(card_vis)

  def evaluate_players_card(self):
      if self.sum_of_cards(self.player_cards) > 20:
        self.is_turn_over = True

  def evaluate_courpiers_card(self):

    if self.sum_of_cards(self.croupier_cards) < 16 :
      print("Courpier needs to pop new card...")
    else:
      self.is_turn_over = True

  def evaluate_game(self): # 0 = winner is courpier , 1 = winner is player

    if self.sum_of_cards(self.player_cards) > 21:
      print("Courpier Wins!")
      self.courpier_wins += 1
      return
    if self.sum_of_cards(self.croupier_cards) > 21:
      print("Player Wins!")
      self.ai_wins += 1
      return
    if  self.sum_of_cards(self.player_cards) > self.sum_of_cards(self.croupier_cards):
      print("Player Wins!")
      self.ai_wins += 1
      return
    else:
      print("Courpier Wins!")
      self.courpier_wins += 1
      return
    
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