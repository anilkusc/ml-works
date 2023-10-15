class AB:
    def __init__(self):
        pass
    def train(self):
        pass

    def move(self,game_board):
        def is_winner(board, player):
            for i in range(3):
                if all(board[i][j] == player for j in range(3)):
                    return True
                if all(board[j][i] == player for j in range(3)):
                    return True
            if all(board[i][i] == player for i in range(3)):
                return True
            if all(board[i][2 - i] == player for i in range(3)):
                return True
            return False

        def is_full(board):
            return all(board[i][j] != 0 for i in range(3) for j in range(3))

        def evaluate(board):
            if is_winner(board, 1):
                return 1
            elif is_winner(board, 2):
                return -1
            else:
                return 0

        def minimax(board, depth, alpha, beta, maximizing_player):
            if is_winner(board, 1):
                return -1
            if is_winner(board, 2):
                return 1
            if is_full(board):
                return 0

            if maximizing_player:
                max_eval = -float('inf')
                best_move = None

                for i in range(3):
                    for j in range(3):
                        if board[i][j] == 0:
                            board[i][j] = 1
                            eval = minimax(board, depth + 1, alpha, beta, False)
                            board[i][j] = 0

                            if eval > max_eval:
                                max_eval = eval
                                best_move = (i, j)
                            alpha = max(alpha, eval)
                            if beta <= alpha:
                                break
                if depth == 0:
                    return best_move
                return max_eval
            else:
                min_eval = float('inf')

                for i in range(3):
                    for j in range(3):
                        if board[i][j] == 0:
                            board[i][j] = 2
                            eval = minimax(board, depth + 1, alpha, beta, True)
                            board[i][j] = 0

                            min_eval = min(min_eval, eval)
                            beta = min(beta, eval)
                            if beta <= alpha:
                                break
                return min_eval

        best_move = minimax(game_board, 0, -float('inf'), float('inf'), True)
        return best_move
