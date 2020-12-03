from typing import Optional, Any

from agents import Agent
from board import Board, DRAW


def printv(_str: Any, verbose: bool):
    if verbose:
        print(_str)


def play_mnk(board: Board, player0: Agent, player1: Agent, verbose: bool = True) -> Optional[str]:
    while True:
        for player in (player0, player1):
            move = player.move()
            board.make_move(move)
            outcome = board.get_outcome()
            if outcome in board.tokens:
                player0.end_game()
                player1.end_game()
                printv(f"{player.name} with token {player.token} wins!", verbose)
                printv(board, verbose)
                return outcome
            elif outcome == DRAW:
                player0.end_game()
                player1.end_game()
                printv(f"Draw!", verbose)
                printv(board, verbose)
                return outcome


def main():
    from agents import Minimax, TabularQLearning, Human
    board = Board(num_rows=3, num_cols=3, num_to_win=3)
    #player1 = FullGameTree(board.tokens, player_idx=0)
    player1 = TabularQLearning(board, player_idx=0)
    player2 = Minimax(board, player_idx=1)
    #player2 = Human(board.tokens, player_idx=1)
    play_mnk(board, player1, player2)


if __name__ == '__main__':
    main()
