from game import Game
from player import Player
player1_type = input("What is player1 type?ab,rl,human: ")
while player1_type != "ab" and player1_type != "human" and player1_type != "rl":
    player1_type = input("Wrong player type.What is player1 type?ab,rl,human: ")
player2_type = input("What is player2 type?ab,rl,human: ")
while player2_type != "ab" and player2_type != "human" and player2_type != "rl":
    player1_type = input("Wrong player type.What is player2 type?ab,rl,human: ")
player1 = Player(AI=player1_type)
player2 = Player(AI=player2_type)
game = Game(player1,player2)
game.start()
