from game import Game
from player import Player
from rl import RL

RL_AI=RL(num_episodes=1000000)
RL_AI.train()
ai_player = Player(AI=RL_AI)
#game = Game(player=Player(AI="human"))
game = Game(player=ai_player)
game.start()
