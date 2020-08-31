from board import Board

WINNING_SETS_3 = [
    {(0, 0), (0, 1), (0, 2)},
    {(1, 0), (1, 1), (1, 2)},
    {(2, 0), (2, 1), (2, 2)},
    {(0, 0), (1, 0), (2, 0)},
    {(0, 1), (1, 1), (2, 1)},
    {(0, 2), (1, 2), (2, 2)},
    {(0, 0), (1, 1), (2, 2)},
    {(0, 2), (1, 1), (2, 0)}
]


def test_init():
    board = Board(num_rows=3, num_cols=3, num_to_win=3)
    assert len(board.winning_sets) == len(WINNING_SETS_3)
    assert all(winning_set in WINNING_SETS_3 for winning_set in board.winning_sets)
    assert len(board.history) == 0


def test_make_child_board():
    board = Board(num_rows=3, num_cols=3, num_to_win=3)
    original_tokens = board.tokens
    child_board = board.make_child_board((0, 0))
    assert board.tokens == original_tokens
    assert len(child_board.history) == 1

