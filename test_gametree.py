from functools import reduce

import pytest

from board import Board, UNDETERMINED
from game_tree import DFSGameTreeNode


def sum_of_falling_factorial(num_spaces, num_to_win) -> int:
    """
    E.g. for num_cols = 3, num_rows = 3, num_to_win = 3,
    we get num_moves = 5 (because 5 moves happen before the end of the game)
    then the possible moves should be [5, 6, 7, 8, 9], so the total sum is:
    1 + 9 + 9 * 8 + 9 * 8 * 7 + 9 * 8 * 7 * 6 + 9 * 8 * 7 * 6 * 5 = 18730
    """
    num_moves = 2 * num_to_win - 1
    possible_moves = list(reversed(range(num_spaces - num_moves + 1, num_spaces + 1)))
    subsequences = [possible_moves[:seq_length + 1] for seq_length in range(len(possible_moves))]
    products = [reduce(lambda x, y: x * y, subsequence) for subsequence in subsequences]
    return 1 + sum(products)


def test_iddfs_game_tree_22():
    num_rows = 2
    num_cols = 2
    num_to_win = 2
    board = Board(num_rows, num_cols, num_to_win)
    root = DFSGameTreeNode(board)
    assert root.heuristic == 0.5
    assert root.outcome is UNDETERMINED
    root.build_tree()
    assert root.outcome == 'X'
    assert root.get_height() == num_to_win * 2
    assert root.get_volume() == sum_of_falling_factorial(num_rows * num_cols, num_to_win)
    assert root.children[(0, 0)].heuristic == 1.

    child_00_01_10 = root.children[(0, 0)].children[(0, 1)].children[(1, 0)]
    child_10_01_00 = root.children[(1, 0)].children[(0, 1)].children[(0, 0)]
    assert child_00_01_10 == child_10_01_00
    assert len(child_00_01_10.parents) == 2


def test_iddfs_game_tree_13():
    num_rows = 1
    num_cols = 3
    num_to_win = 2
    board = Board(num_rows, num_cols, num_to_win)
    root = DFSGameTreeNode(board)
    assert root.outcome is UNDETERMINED
    root.build_tree()
    assert root.outcome == 'X'
    assert root.children[(0, 1)].heuristic == 1.
    assert root.children[(0, 0)].heuristic == 0.75
    assert root.get_height() == 2 * num_to_win
    assert root.get_volume() == sum_of_falling_factorial(num_rows * num_cols, num_to_win)


def test_iddfs_game_tree_23():
    num_rows = 2
    num_cols = 3
    num_to_win = 2
    board = Board(num_rows, num_cols, num_to_win)
    root = DFSGameTreeNode(board)
    assert root.outcome is UNDETERMINED
    root.build_tree()
    assert root.outcome == 'X'
    assert root.children[(0, 1)].heuristic == 1.
    assert root.children[(0, 0)].heuristic == 0.8
    assert root.get_height() == 2 * num_to_win
    assert root.get_volume() == sum_of_falling_factorial(num_rows * num_cols, num_to_win)


def test_iddfs_game_tree_33():
    num_rows = 3
    num_cols = 3
    num_to_win = 3
    board = Board(num_rows, num_cols, num_to_win)
    root = DFSGameTreeNode(board, filter_heuristic=False)
    assert root.outcome is UNDETERMINED
    root.build_tree()
    assert root.outcome == 'X'
    assert root.get_height() == 2 * num_to_win
    assert root.get_volume() == sum_of_falling_factorial(num_rows * num_cols, num_to_win)

    #########
    # 1 Ply #
    #########
    child_11 = root.children[(1, 1)]
    child_11.build_tree()
    assert child_11.heuristic > 0.5
    assert child_11.outcome == 'X'

    child_00 = root.children[(0, 0)]
    child_00.build_tree()
    assert child_00.heuristic > 0.5
    assert child_00.outcome == 'X'

    child_01 = root.children[(0, 1)]
    child_01.build_tree()
    assert child_01.heuristic < 0.5
    assert child_01.outcome == 'X'

    # In order from better move to worse move
    assert child_11.heuristic > child_00.heuristic > child_01.heuristic

    #########
    # 2 Ply #
    #########
    child_11_01 = root.children[(1, 1)].children[(0, 1)]
    child_11_01.build_tree()
    assert child_11_01.heuristic < 0.5

    child_11_00 = root.children[(1, 1)].children[(0, 0)]
    child_11_00.build_tree()
    assert child_11_00.heuristic < 0.5

    assert child_11_00.heuristic > child_11_01.heuristic

    #########
    # 3 Ply #
    #########
    child_11_01_22 = root.children[(1, 1)].children[(0, 1)].children[(2, 2)]
    child_11_01_22.build_tree()
    assert child_11_01_22.heuristic > 0.5

    child_11_00_22 = root.children[(1, 1)].children[(0, 0)].children[(2, 2)]
    child_11_00_22.build_tree()
    assert child_11_00_22.heuristic > 0.5

    assert child_11_01_22.heuristic > child_11_00_22.heuristic

    #########
    # 4 Ply #
    #########
    child_11_01_22_10 = root.children[(1, 1)].children[(0, 1)].children[(2, 2)].children[(1, 0)]
    child_11_01_22_10.build_tree()
    assert child_11_01_22_10.heuristic == 0.

    child_11_00_22_10 = root.children[(1, 1)].children[(0, 0)].children[(2, 2)].children[(1, 0)]
    child_11_00_22_10.build_tree()
    assert pytest.approx(child_11_00_22_10.heuristic) == 0.05

    assert child_11_00_22_10.heuristic > child_11_01_22_10.heuristic
