from helpers import qlearning,initialize_game,play_game

env,q_table = initialize_game()

q_table = qlearning(env,q_table,num_episodes=5000,learning_rate=0.1,discount_rate=0.9)

play_game(env,q_table)

env.close()