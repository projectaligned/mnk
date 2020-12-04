import time
from typing import Type

from agents import Random, InOrder, Agent, TabularQLearning, DeepQLearning
from mnk import play_mnk
from board import DRAW, Board
import matplotlib.pyplot as plt


def compare_players(player1_type: Type[Agent], player2_type: Type[Agent], board: Board, num_rounds: int):
    outcomes = []
    for _ in range(num_rounds):
        board_clone = board.clone()
        player1 = player1_type(board_clone, player_idx=0)
        player2 = player2_type(board_clone, player_idx=1)
        outcomes.append(play_mnk(board_clone, player1, player2, verbose=False))

    player1_wins = len([w for w in outcomes if w == player1_type.name])
    player2_wins = len([w for w in outcomes if w == player2_type.name])
    draws = len([w for w in outcomes if w is None])
    print(f'{num_rounds} games played.')
    print(f'player1 won {float(player1_wins) / num_rounds * 100.:.2f} of the games')
    print(f'player2 won {float(player2_wins) / num_rounds * 100.:.2f} of the games')
    print(f'{float(draws) / num_rounds * 100.:.2f} of the games ended in a draw')


def main():
    board = Board(num_rows=3, num_cols=3, num_to_win=3)
    t1 = time.perf_counter()
    compare_players(TabularQLearning, Random, board, 100)
    t2 = time.perf_counter()
    print(t2 - t1)


if __name__ == '__main__':
    main()
    # Make a tournament of games
    # See which strategies are non-dominated
    # ai_in_order (100%) vs. ai_in_order (0%) vs. draw (0%)
    # ai_in_order (78%) vs. ai_random (18%) vs. draw (4%)
    # ai_random (52%) vs. ai_in_order (44%) vs. draw (4%)
    # ai_random (58%) vs. ai_random (29%) vs. draw (13%)

