import pytest

from agents import InOrder, Random, OneStep, OneStepBlocking, Minimax, TabularQLearning
from board import Board


@pytest.fixture()
def almost_win_board():
    board = Board(num_rows=3, num_cols=3, num_to_win=3)
    board.make_move((1, 1))
    board.make_move((0, 1))
    board.make_move((0, 0))
    board.make_move((1, 0))
    return board


@pytest.fixture()
def almost_lose_board():
    board = Board(num_rows=3, num_cols=3, num_to_win=3)
    board.make_move((0, 0))
    board.make_move((2, 0))
    board.make_move((1, 2))
    board.make_move((0, 2))
    return board


def test_in_order_valid(almost_win_board):
    ai = InOrder(almost_win_board, player_idx=0)
    move = ai.move()
    assert almost_win_board.is_move_free(move)


def test_random_valid(almost_win_board):
    ai = Random(almost_win_board, player_idx=0)
    move = ai.move()
    assert almost_win_board.is_move_free(move)


def test_one_step_win(almost_win_board):
    ai = OneStep(almost_win_board, player_idx=0)
    move = ai.move()
    assert move == (2, 2)


def test_one_step_blocking_win(almost_win_board):
    ai = OneStepBlocking(almost_win_board, player_idx=0)
    move = ai.move()
    assert move == (2, 2)


def test_one_step_blocking_lose(almost_lose_board):
    ai = OneStepBlocking(almost_lose_board, player_idx=0)
    move = ai.move()
    assert move == (1, 1)


def test_minimax_win(almost_win_board):
    ai = Minimax(almost_win_board, player_idx=0)
    move = ai.move()
    assert move == (2, 2)


def test_minimax_lose(almost_lose_board):
    ai = Minimax(almost_lose_board, player_idx=0)
    move = ai.move()
    assert move == (1, 1)


def test_minimax_sequence():
    board = Board(num_rows=3, num_cols=3, num_to_win=3)
    player0 = Minimax(board, player_idx=0)
    player1 = Minimax(board, player_idx=1)
    board.make_move(player0.move())
    board.make_move(player1.move())
    board.make_move(player0.move())
    assert len(board.history) == 3


def test_q_learning_valid(almost_win_board):
    ai = TabularQLearning(almost_win_board, player_idx=0)
    move = ai.move()
    assert almost_win_board.is_move_free(move)
