
class Player:
  def __init__(self, AI):# human,ab,rl
    self.win = 0
    self.AI = AI

  def is_hit(self): # Take another card = hit
    if self.AI == "human":
      hit_or_stand = ""
      while not (hit_or_stand == "h" or hit_or_stand == "s"):
        hit_or_stand = input("Do you want to hit or stand?h/s \n")
        if hit_or_stand == "h":
          return True
        elif hit_or_stand == "s":
          return False
        else:
          print("Invalid option. Please select again.")
    else:
      return self.AI.move()
