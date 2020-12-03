import time
from typing import Optional, Any

from agents import Agent
from board import Board, DRAW


def printv(_str: Any, verbose: bool):
    if verbose:
        print(_str)


def play_mnk(board: Board, player0: Agent, player1: Agent, verbose: bool = True) -> Optional[str]:
    while True:
        for player in (player0, player1):
            #print(f"{player.name} Making Move")
            t1 = time.perf_counter()
            move = player.move()
            t2 = time.perf_counter()
            #print(t2 - t1)
            board.make_move(move)
            #print(f"Checking Outcome")
            t1 = time.perf_counter()
            outcome = board.get_outcome()
            t2 = time.perf_counter()
            #print(t2 - t1)
            if outcome in board.tokens:
                player0.end_game()
                player1.end_game()
                #printv(f"{player.name} with token {player.token} wins!", verbose)
                #printv(board, verbose)
                return outcome
            elif outcome == DRAW:
                player0.end_game()
                player1.end_game()
                #printv(f"Draw!", verbose)
                #printv(board, verbose)
                return outcome


def main():
    from agents import Minimax, TabularQLearning, Human, InOrder, DeepQLearning, Random
    board = Board(num_rows=5, num_cols=5, num_to_win=5)
    #player1 = FullGameTree(board.tokens, player_idx=0)
    player1 = DeepQLearning(board, player_idx=0)
    player2 = Random(board, player_idx=1)
    #player2 = Minimax(board, player_idx=1)
    #player2 = Human(board.tokens, player_idx=1)
    #print(f"Playing the game")
    t1 = time.perf_counter()
    play_mnk(board, player1, player2)
    t2 = time.perf_counter()
    print(t2-t1)


if __name__ == '__main__':
    main()
