from game import Game
from player import Player
from rl import RL
from abprunning import AB

player1_type = input("What is player1 type?ab,rl,human: ")
while player1_type != "ab" and player1_type != "human" and player1_type != "rl":
    player1_type = input("Wrong player type.What is player1 type?ab,rl,human: ")
player2_type = input("What is player2 type?ab,rl,human: ")
while player2_type != "ab" and player2_type != "human" and player2_type != "rl":
    player1_type = input("Wrong player type.What is player2 type?ab,rl,human: ")

if player1_type=="rl":
    player1AI = RL()
    print("Player 1 AI is training. Please wait...")
    player1AI.train()
    print("Completed")
elif player1_type=="ab":
    player1AI = AB()
    print("Player 1 AI is training. Please wait...")
else:
    player1AI = "human"

if player2_type=="rl":
    player2AI = RL()
    print("Player 2 AI is training. Please wait...")
    player2AI.train()
    print("Completed")
elif player2_type=="ab":
    player2AI = AB()
    print("Player 2 AI is training. Please wait...")
else:
    player2AI = "human"

player1 = Player(AI=player1AI)
player2 = Player(AI=player2AI)
game = Game(player1,player2)
game.start()
