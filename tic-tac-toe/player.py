
class Player:
  def __init__(self, AI):# human,ab,rl
    self.score = 0
    self.AI = AI

  def move(self,game_board):
    if self.AI == "human":
      x = int(input("please enter column: "))
      y = int(input("please enter row: "))
      return x,y
    else:
      return self.AI.move(game_board)