from game import Game
from player import Player
from rl import RL

game = Game(player=Player(AI="human"))
game.start()
