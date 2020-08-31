import matplotlib.pyplot as plt

from agents import TabularQLearning, Random
from board import Board, DRAW, TOKENS
from mnk import play_mnk

PLOT_DELAY = 50


def train_q_learning(board: Board, num_rounds: int) -> None:
    outcome_counts = {DRAW: 0, TOKENS[0]: 0, TOKENS[1]: 0}
    percentages = {DRAW: [], TOKENS[0]: [], TOKENS[1]: []}
    q_table = None
    for idx in range(num_rounds):
        board_clone = board.clone()
        random_agent = Random(board_clone, player_idx=1)
        q_learning_agent = TabularQLearning(board_clone, player_idx=0, q_table=q_table, ignorance_bias=0.1)
        outcome = play_mnk(board_clone, q_learning_agent, random_agent, verbose=False)
        outcome_counts[outcome] += 1
        q_table = q_learning_agent.q_table

        total = sum(outcome_counts.values())
        percentages[DRAW].append(outcome_counts[DRAW] / total)
        percentages[TOKENS[0]].append(outcome_counts[TOKENS[0]] / total)
        percentages[TOKENS[1]].append(outcome_counts[TOKENS[1]] / total)

    x_axis = list(range(num_rounds))[PLOT_DELAY:]
    plt.plot(x_axis, percentages[DRAW][PLOT_DELAY:], label='DRAW')
    plt.plot(x_axis, percentages[TOKENS[0]][PLOT_DELAY:], label=f'{TOKENS[0]}')
    plt.plot(x_axis, percentages[TOKENS[1]][PLOT_DELAY:], label=f'{TOKENS[1]}')
    plt.legend()
    plt.show()


def main():
    board = Board(num_rows=3, num_cols=3, num_to_win=3)
    train_q_learning(board, num_rounds=2000)


if __name__ == '__main__':
    main()
