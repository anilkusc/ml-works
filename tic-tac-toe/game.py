class Game:
  def __init__(self, player1, player2):
    self.player1 = player1
    self.player2 = player2
    self.turn = 1
    self.game_board = [[0,0,0],[0,0,0],[0,0,0]]
  
  def start(self):
    self.move = 1
    while True:
      print("who's turn:"+ str(self.turn))
      print("move:"+ str(self.move))
      self.render_board()
      if self.turn == 1:
        x,y = self.player1.move(self.game_board)
      else:
        x,y = self.player2.move(self.game_board)
      while not self.is_move_valid(x,y):
        print("invalid move. please try again.")
        if self.turn == 1:
          x,y = self.player1.move(self.game_board)
        else:
          x,y = self.player2.move(self.game_board)
      print("x:",x)
      print("y:",y)
      self.game_board[x][y] = self.turn
      if self.is_winner():
        print("winner is player: " + str(self.turn))
        if self.turn == 1:
          self.player1.score += 1
        else:
          self.player2.score += 1
        self.render_board()
        self.restart_game()
        continue
      if self.is_draw():
        print("this is a draw!")
        self.player1.score += 1
        self.player2.score += 1        
        self.render_board()
        self.restart_game()
      if self.turn == 1:
        self.turn = 2
      else:
        self.turn = 1
      self.move += 1

  def restart_game(self):
    print("game is restarting...")
    self.turn = 1
    self.game_board = [[0,0,0],[0,0,0],[0,0,0]]
    self.move = 1
    print("#####################################")

  def render_board(self):
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
    if 0 > x or 2 < x or 0 > y or 2 < y: 
      return False
    if self.game_board[x][y] != 0:
      return False    
    return True

  def is_winner(self):
    for row in self.game_board:
      # control rows
      if row[0] == row[1] == row[2] and row[0] > 0 and row[1] > 0 and row[2] > 0:
        return True
    for i,_ in enumerate(self.game_board):
      # control columns  y,x
      if self.game_board[0][i] == self.game_board[1][i] == self.game_board[2][i] and self.game_board[0][i] > 0 and self.game_board[1][i] > 0 and self.game_board[2][i] > 0:
        return True
    if self.game_board[0][0] == self.game_board[1][1] == self.game_board[2][2] and self.game_board[0][0] > 0 and self.game_board[1][1] > 0 and self.game_board[2][2] > 0:
      return True
    if self.game_board[0][2] == self.game_board[1][1] == self.game_board[2][0] and self.game_board[0][2] > 0 and self.game_board[1][1] > 0 and self.game_board[2][0] > 0:
      return True
    return False

  def is_draw(self):
    for row in self.game_board:
      for item in row:
        if item == 0:
          return False
    return True
