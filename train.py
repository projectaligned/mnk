import matplotlib.pyplot as plt

from agents import TabularQLearning, InOrder, Agent, OneStep, OneStepBlocking, Minimax, Random
from board import Board
from mnk import play_mnk

PLOT_DELAY = 50


def train_q_learner(board: Board, num_rounds: int,
                    ignorance_bias: float = 0.1,
                    discount_rate: float = 0.95,
                    learning_rate: float = 0.4,
                    title: str = '') -> dict:
    """
    Train the Tabular_Q_Learning Agent against the Random bot until it reaches convergence
    """
    labels = [None, TabularQLearning.name, Random.name]
    outcome_counts = {label: 0 for label in labels}
    percentages = {label: [] for label in labels}
    q_table = None
    for idx in range(num_rounds):
        board_clone = board.clone()
        q_learning_agent = TabularQLearning(board_clone, player_idx=0, q_table=q_table,
                                            ignorance_bias=ignorance_bias,
                                            discount_rate=discount_rate,
                                            learning_rate=learning_rate,
                                            counter=idx,
                                            max_counter=num_rounds)
        random_agent = Random(board_clone, player_idx=1)
        outcome = play_mnk(board_clone, q_learning_agent, random_agent, verbose=False)
        outcome_counts[outcome] += 1
        q_table = q_learning_agent.q_table

        total = sum(outcome_counts.values())
        for label in labels:
            percentages[label].append(outcome_counts[label] / total)

    x_axis = list(range(num_rounds))[PLOT_DELAY:]
    for label in labels:
        plt.plot(x_axis, percentages[label][PLOT_DELAY:], label=label)
    plt.title(title)
    plt.legend()
    plt.show()
    return percentages


def main():
    board = Board(num_rows=3, num_cols=3, num_to_win=3)
    train_q_learner(board, num_rounds=10000)


if __name__ == '__main__':
    main()
