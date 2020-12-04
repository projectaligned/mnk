import matplotlib.pyplot as plt

from agents import TabularQLearning, InOrder, Agent, OneStep, OneStepBlocking, Minimax
from board import Board
from mnk import play_mnk

PLOT_DELAY = 50


def train(board: Board, num_rounds: int, player_one: type(Agent), player_two: type(Agent)) -> None:
    """
    Train the Tabular_Q_Learning Agent against the Random bot until it reaches convergence
    """
    labels = [None, player_one.name, player_two.name]
    outcome_counts = {label: 0 for label in labels}
    percentages = {label: [] for label in labels}
    q_table = None
    for idx in range(num_rounds):
        board_clone = board.clone()
        q_learning_agent = player_one(board_clone, player_idx=0, q_table=q_table, ignorance_bias=0.1)
        random_agent = player_two(board_clone, player_idx=1)
        outcome = play_mnk(board_clone, q_learning_agent, random_agent, verbose=False)
        outcome_counts[outcome] += 1
        q_table = q_learning_agent.q_table

        total = sum(outcome_counts.values())
        for label in labels:
            percentages[label].append(outcome_counts[label] / total)

    x_axis = list(range(num_rounds))[PLOT_DELAY:]
    for label in labels:
        plt.plot(x_axis, percentages[label][PLOT_DELAY:], label=label)
    plt.legend()
    plt.show()


def main():
    board = Board(num_rows=3, num_cols=3, num_to_win=3)
    train(board, num_rounds=1000, player_one=TabularQLearning, player_two=Minimax)


if __name__ == '__main__':
    main()
