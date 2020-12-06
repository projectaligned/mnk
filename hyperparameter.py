import itertools

from agents import TabularQLearning
from board import Board
from train import train_q_learner


def search_tabular_q_learning():
    discount_rates = [0.9, 0.95, 0.99]
    ignorance_biases = [0, 0.1]
    learning_rates = [0.1, 0.3, 0.5]
    board = Board(num_rows=3, num_cols=3, num_to_win=3)
    grid = list(itertools.product(discount_rates, learning_rates, ignorance_biases))
    for (discount_rate, learning_rate, ignorance_bias) in grid:
        title = f"discount={discount_rate}, learning={learning_rate}, ignorance={ignorance_bias}"
        percentages = train_q_learner(board,
                                      num_rounds=1000,
                                      ignorance_bias=ignorance_bias,
                                      discount_rate=discount_rate,
                                      learning_rate=learning_rate,
                                      title=title)
        final_win_percentage = percentages[TabularQLearning.name][-1]
        print(f"final win percentage = {final_win_percentage} for {title}")


def check_stability():
    discount_rate = 0.9
    learning_rate = 0.3
    ignorance_bias = 0.1
    board = Board(num_rows=3, num_cols=3, num_to_win=3)
    title = f"discount={discount_rate}, learning={learning_rate}, ignorance={ignorance_bias}"
    final_percentages = []
    for iteration in range(5):
        percentages = train_q_learner(board,
                                      num_rounds=1000,
                                      ignorance_bias=ignorance_bias,
                                      discount_rate=discount_rate,
                                      learning_rate=learning_rate,
                                      title=title)
        final_win_percentage = percentages[TabularQLearning.name][-1]
        print(f"final win percentage = {final_win_percentage} for {title}")


if __name__ == "__main__":
    #search_tabular_q_learning()
    check_stability()
