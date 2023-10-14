class Player:
  def __init__(self, AI):
    self.score = 0
    self.AI = AI # human,ab,rl
  def move(self):
    x = int(input("please enter x coordinate: "))
    y = int(input("please enter y coordinate: "))
    return x,y
