import json
import math
import os
import random
from abc import ABC
from collections import defaultdict
from typing import Optional, DefaultDict, List

import tensorflow as tf
from tensorflow import keras

from board import Board, Move, DRAW, UNDETERMINED
from game_tree import DFSGameTreeNode


class Agent(ABC):

    name = 'AbstractAgent'

    def __init__(self, board: Board, player_idx: int):
        self.player_idx = player_idx
        self.board = board
        self.token = self.board.tokens[player_idx]
        self.opponent_token = self.board.tokens[1 - player_idx]

    def move(self) -> Move:
        raise NotImplementedError

    def state(self) -> Optional[dict]:
        pass

    def end_game(self) -> None:
        pass


def get_valid_input(prefix: str, min_val: int, max_val: int) -> int:
    value = input(f"Enter {prefix} value: ")
    try:
        value = int(value)
    except (TypeError, ValueError):
        print(f"The value {value} could not be cast to an integer")
        return get_valid_input(prefix, min_val, max_val)

    if not min_val <= value <= max_val:
        print(f"The value {value} must be in [{min_val}, {max_val}]")
        return get_valid_input(prefix, min_val, max_val)
    else:
        return value


class Human(Agent):

    name = "Human"

    def move(self):
        print(self.board)
        row_idx = get_valid_input('row', 0, self.board.num_rows - 1)
        col_idx = get_valid_input('col', 0, self.board.num_cols - 1)
        move = (row_idx, col_idx)
        if self.board.is_move_free(move):
            return move
        else:
            return self.move()


class InOrder(Agent):

    name = "InOrder"

    def move(self) -> Move:
        return self.board.get_available_moves()[0]


class Random(Agent):

    name = "Random"

    def move(self) -> Move:
        row_idx = random.randint(0, self.board.num_rows - 1)
        col_idx = random.randint(0, self.board.num_cols - 1)
        move = (row_idx, col_idx)
        if self.board.is_move_free(move):
            return move
        else:
            return self.move()


class OneStep(Agent):

    name = "OneStep"

    def move(self) -> Move:
        moves = self.board.get_available_moves()
        for move in moves:
            if self.board.does_move_win(move, self.token):
                return move
        return Random(self.board, self.player_idx).move()


class OneStepBlocking(Agent):

    name = "OneStepBlocking"

    def move(self) -> Move:
        moves = self.board.get_available_moves()
        for move in moves:
            if self.board.does_move_win(move, self.token):
                return move
        for move in moves:
            if self.board.does_move_win(move, self.opponent_token):
                return move
        return Random(self.board, self.player_idx).move()


class Minimax(Agent):

    name = "Minimax"

    def __init__(self, board: Board, player_idx: int, lookahead_depth: Optional[int] = None, ab_prune: bool = False):
        """
        Uses the minimax algorithm with alpha-beta pruning
        """
        super().__init__(board, player_idx)
        self.active_node: Optional[DFSGameTreeNode] = DFSGameTreeNode(self.board.clone(), ab_prune=ab_prune)
        self.lookahead_depth = lookahead_depth
        self.ab_prune = ab_prune

    def update_active_node(self) -> None:
        new_moves = self.board.history[len(self.active_node.board.history):]
        for move in new_moves:
            if move not in self.active_node.children:
                self.active_node.build_tree(self.lookahead_depth)
            self.active_node = self.active_node.children[move]

    def move(self) -> Move:
        self.update_active_node()
        self.active_node.build_tree(self.lookahead_depth)

        outcome_preference_order = [self.token, DRAW, UNDETERMINED, self.opponent_token]
        moves_sets = [
            (outcome, [(m, child) for (m, child) in self.active_node.children.items() if child.outcome == outcome])
            for outcome in outcome_preference_order]
        for outcome, moves in moves_sets:
            if len(moves) > 0:
                heuristics = [child.heuristic for (_, child) in moves]
                best_heuristic = max(heuristics)
                best_moves = [(move, child) for (move, child) in moves if child.heuristic == best_heuristic]
                # want the slowest outcome if the opponent will win
                if outcome == self.opponent_token:
                    best_move, _ = max(best_moves, key=lambda p: p[1].height)
                # want the fastest outcome otherwise
                else:
                    best_move, _ = min(best_moves, key=lambda p: p[1].height)
                return best_move

    def end_game(self):
        self.active_node = DFSGameTreeNode(self.board.clone(), ab_prune=self.ab_prune)


QTable = DefaultDict[int, float]


class TabularQLearning(Agent):
    """
    Optionally performs Double Q-Learning if selected.
    """

    name = "TabularQLearning"

    def __init__(self, board: Board, player_idx: int,
                 discount_rate: float = 0.95,
                 learning_rate: float = 0.1,
                 ignorance_bias: float = 0.1,
                 q_table_a: Optional[QTable] = None,
                 q_table_b: Optional[QTable] = None,
                 use_double_q_table: bool = True,
                 counter: Optional[int] = None,
                 max_counter: Optional[int] = None):
        super().__init__(board, player_idx)
        self.outcome_value = {self.token: 1., DRAW: 0., UNDETERMINED: 0., self.opponent_token: -1.}
        self.ignorance_bias = ignorance_bias
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        # The greediness should start high, near 1, and then decrease to zero over the course of the trials
        if counter and max_counter:
            self.epsilon_greedy = 0.5 * (1 - math.ceil(float(counter+1) / float(max_counter) * 10) / 10)
        else:
            self.epsilon_greedy = 0.1
        self.q_table_a: QTable = q_table_a or self.load_q_table('a') or self.build_q_table()
        self.use_double_q_table = use_double_q_table
        if self.use_double_q_table:
            self.q_table_b: QTable = q_table_b or self.load_q_table('b') or self.build_q_table()

    def get_file_name(self) -> str:
        return self.name + '.json'

    def get_table_name(self, suffix: str) -> str:
        return "_".join(
            [str(self.board.num_rows), str(self.board.num_cols), str(self.board.num_to_win), suffix]
                        )

    def load_q_table(self, suffix: str) -> Optional[QTable]:
        if not os.path.exists(self.get_file_name()):
            with open(self.get_file_name(), 'w') as wfp:
                json.dump({}, wfp)
            return None

        with open(self.get_file_name(), 'r') as rfp:
            loaded_state = json.load(rfp)
            table_name = self.get_table_name(suffix)
            loaded_table = loaded_state.get(table_name, dict())
            q_table = defaultdict(lambda: self.outcome_value[UNDETERMINED] + self.ignorance_bias)
            q_table.update(loaded_table)
            return q_table

    def save_q_table(self, suffix: str) -> None:
        with open(self.get_file_name(), 'r') as rfp:
            loaded_state = json.load(rfp)

        table_name = self.get_table_name(suffix)
        loaded_table = loaded_state.get(table_name, {})
        if suffix == 'a':
            loaded_table.update(self.q_table_a)
        if suffix == 'b':
            loaded_table.update(self.q_table_b)
        loaded_state[table_name] = loaded_table

        with open(self.get_file_name(), 'w') as wfp:
            json.dump(loaded_state, wfp)

    def increment_games_played(self, suffix: str) -> None:
        with open(self.get_file_name(), 'r+') as fp:
            loaded_state = json.load(fp)

        games_played_label = self.get_table_name(suffix) + "_games_played"
        games_played = loaded_state.get(games_played_label, 0)
        loaded_state[games_played_label] = games_played + 1

        with open(self.get_file_name(), 'w') as wfp:
            json.dump(loaded_state, wfp)

    def state(self) -> str:
        with open(self.get_file_name(), 'r') as fp:
            loaded_state = json.load(fp)

        # since I am sharing a table for single and double q learning, I am just going to store the total number of
        # games played in 'a'
        label = self.get_table_name('a') + "_games_played"
        return json.dumps({label: loaded_state[label],
                           'discount_rate': self.discount_rate,
                           'learning_rate': self.learning_rate,
                           'ignorance_bias': self.ignorance_bias})

    def build_q_table(self) -> QTable:
        return defaultdict(lambda: self.outcome_value[UNDETERMINED] + self.ignorance_bias)

    def move(self) -> Move:
        # explore
        all_moves = self.board.get_available_moves()
        if random.random() > self.epsilon_greedy:
            return random.sample(all_moves, k=1)[0]

        # exploit
        if self.use_double_q_table:
            scored_moves = [(move, sum([self.q_table_a[hash(board)], self.q_table_b[hash(board)]]) / 2.)
                            for (move, board) in self.board.get_child_boards()]
        else:
            scored_moves = [(move, self.q_table_a[hash(board)]) for (move, board) in self.board.get_child_boards()]
        best_score = max([score for (_, score) in scored_moves])
        best_moves = [move for (move, score) in scored_moves if score >= best_score]
        return random.sample(best_moves, k=1)[0]

    def update_q_table(self, intermediate_boards: List[Board], target_table: QTable, reference_table: QTable) -> None:
        target_table[hash(self.board)] = self.outcome_value[self.board.get_outcome()]
        for move_idx, board in reversed(list(enumerate(intermediate_boards))):
            best_child_score = max([reference_table[hash(board)] for (_, board) in board.get_child_boards()])
            target_table[hash(board)] = (
                    target_table[hash(board)] +
                    self.learning_rate * (self.learning_rate * self.discount_rate * best_child_score -
                                          target_table[hash(board)]))

    def end_game(self) -> None:

        active_board = Board(self.board.num_rows, self.board.num_cols, self.board.num_to_win, self.board.winning_sets)
        intermediate_boards = [active_board]
        # the moves are stored in the history from early to late
        # but we want to iterate over the boards from late to early so that we can update the earlier board score based
        # on the later board scores.
        for move in self.board.history[:-1]:
            active_board = active_board.make_child_board(move)
            intermediate_boards.append(active_board)

        if self.use_double_q_table:
            if bool(random.getrandbits(1)):
                self.update_q_table(intermediate_boards, self.q_table_a, self.q_table_b)
            else:
                self.update_q_table(intermediate_boards, self.q_table_b, self.q_table_a)
        else:
            self.update_q_table(intermediate_boards, self.q_table_a, self.q_table_a)

        self.increment_games_played('a')
        self.save_q_table('a')
        if self.use_double_q_table:
            self.save_q_table('b')
            self.increment_games_played('b')


class DeepQLearning(Agent):

    name = "DeepQLearning"

    architecture = 'dqn'

    def __init__(self, board: Board, player_idx: int, discount_rate: float = 0.95, learning_rate: float = 0.9,
                 ignorance_bias: float = 0.1, num_hidden_units: int = 16, dqn: Optional[keras.Model] = None):
        super().__init__(board, player_idx)
        self.outcome_value = {self.token: 1., DRAW: 0., UNDETERMINED: 0., self.opponent_token: -1.}
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.ignorance_bias = ignorance_bias
        self.input_shape = self.board.num_rows * self.board.num_cols * 3  # tokens + blank
        self.num_hidden_units = num_hidden_units
        self.model: keras.Model = dqn or self.load_model() or self.build_model()

    def get_file_name(self) -> str:
        return "_".join(
            [str(self.board.num_rows), str(self.board.num_cols), str(self.board.num_to_win), self.architecture])

    def load_model(self) -> Optional[keras.Model]:
        if os.path.exists(self.get_file_name() + '.h5'):
            return keras.models.load_model(self.get_file_name() + '.h5')
        elif os.path.exists(self.get_file_name() + '.ckpt'):
            model = self.build_model()
            model.load_weights(self.get_file_name() + '.ckpt')
            return model

    def save_model(self) -> None:
        self.model.save(self.get_file_name() + '.h5')

    def build_model(self) -> keras.Model:
        model = keras.Sequential([
            keras.layers.Dense(self.num_hidden_units, activation=tf.nn.relu,
                               kernel_regularizer=keras.regularizers.l2(0.01),
                               input_shape=self.input_shape),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def move(self) -> Move:
        scored_moves = [(move, self.model.predict(board.one_hot_encode()))
                        for (move, board) in self.board.get_child_boards()]
        best_score = max([score for (_, score) in scored_moves])
        best_moves = [move for (move, score) in scored_moves if score >= best_score]
        return random.sample(best_moves, k=1)[0]

    def end_game(self) -> None:
        cp_callback = tf.keras.callbacks.ModelCheckpoint(self.get_file_name() + '.ckpt',
                                                         save_weights_only=True, verbose=1)
        self.model.fit(self.board.one_hot_encode(), self.outcome_value[self.board.get_outcome()],
                       callbacks=[cp_callback])

        active_board = Board(self.board.num_rows, self.board.num_cols, self.board.num_to_win, self.board.winning_sets)
        intermediate_boards = [active_board]
        for move in self.board.history[:-1]:
            active_board = active_board.make_child_board(move)
            intermediate_boards.append(active_board)

        for board in reversed(intermediate_boards):
            best_child_score = max([self.model.predict(board.one_hot_encode()) for (_, board)
                                    in board.get_child_boards()])
            score = ((1 - self.learning_rate) * self.model.predict(board.one_hot_encode()) +
                     self.learning_rate * self.discount_rate * best_child_score)
            self.model.fit(board.one_hot_encode(), score, callbacks=[cp_callback])
        self.save_model()

