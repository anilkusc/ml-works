class Game:
  def __init__(self, player1, player2):
    self.player1 = player1
    self.player2 = player2
    self.game_board = [[0,0,0],[0,0,0],[0,0,0]]
  
  def render_boards(self):
    print("_____________")
    print("| "+self.render_board_item(self.game_board[0][0])+" | "+self.render_board_item(self.game_board[0][1])+" | "+self.render_board_item(self.game_board[0][2])+" |")
    print("| "+self.render_board_item(self.game_board[1][0])+" | "+self.render_board_item(self.game_board[1][1])+" | "+self.render_board_item(self.game_board[1][2])+" |")
    print("| "+self.render_board_item(self.game_board[2][0])+" | "+self.render_board_item(self.game_board[2][1])+" | "+self.render_board_item(self.game_board[2][2])+" |")
    print("-------------")
    
  def render_board_item(self,number):
    if number == 0:
      return " "
    elif number == 1:
      return "X"
    elif number == 2:
      return "O"

  def is_move_valid(self,x,y):
    if 0 > x or 3 < x or 0 > y or 3 < y: 
      return False
    if self.game_board[x][y] != 0:
      return False    
    return True
