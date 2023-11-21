import random

class Game:
  def __init__(self,player):

    self.cards = self.shuffle_cards()
    self.croupier_cards = []
    self.player_cards = []
    self.is_turn_over = False
    self.player = player

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
    while True:
      print("#############################################")
      print("Starting new turn")
      self.turn()
      self.is_turn_over = False
      if len(self.cards) < 10:
        print("#############################################")
        print("Reshufling cards")
        self.cards = self.shuffle_cards()

  def turn(self):
    self.deliver_first_time()
    self.print_current_status_censored()
    
    while not self.is_turn_over and self.player.is_hit():
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
    for i,card in enumerate(self.croupier_cards):
      if len(self.croupier_cards)-1 == i:
        print("# ? #")
      else:
        print("# "+str(card)+" #")

    print("---------player's hand---------")
    for card in self.player_cards:
      print("# "+str(card)+" #")

  def print_current_status(self):
    print("---------croupier's hand---------")
    for i,card in enumerate(self.croupier_cards):
        print("# "+str(card)+" #")
    
    print("---------player's hand---------")
    for card in self.player_cards:
      print("# "+str(card)+" #")

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
      self.player.win = 0
      return
    if self.sum_of_cards(self.croupier_cards) > 21:
      print("Player Wins!")
      self.player.win = 1
      return
    if  self.sum_of_cards(self.player_cards) > self.sum_of_cards(self.croupier_cards):
      print("Player Wins!")
      self.player.win = 1
      return
    else:
      print("Courpier Wins!")
      self.player.win = 0
      return
    
  def sum_of_cards(self,cards):
    total = 0
    for card in cards:
      if card != 1:
        total = total + card
      else:
        if (total+card) > 21:
          total = total + 1
        else:
          total = total + 11
    return total