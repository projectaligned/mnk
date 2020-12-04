import json
from typing import List, Set, Tuple, Optional

BLANK = '-'
DRAW = '='
UNDETERMINED = '?'
TOKENS = ['X', 'O']

Move = Tuple[int, int]
State = List[str]
WinningSets = List[Set[Move]]


class Board:

    def __init__(self, num_rows: int, num_cols: int, num_to_win: int, winning_sets: Optional[WinningSets] = None):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_to_win = num_to_win
        self.__state: State = [BLANK for _ in range(num_rows * num_cols)]
        self.winning_sets = winning_sets or get_winning_sets(num_rows, num_cols, num_to_win)
        self.tokens = TOKENS
        self.history: List[Move] = []
        self.player_moves = {TOKENS[0]: set(), TOKENS[1]: set()}

    def __str__(self) -> str:
        idxs = range(0, self.num_rows * (self.num_cols + 1), self.num_cols)
        pairs = list(zip(idxs[:-1], idxs[1:]))
        rows = [self.__state[start: end] for (start, end) in pairs]
        return '\n'.join(['|'.join(row) for row in rows])

    def __hash__(self):
        return hash(tuple(self.__state))

    def one_hot_encode(self) -> List[int]:
        def move_map(token: str) -> List[int]:
            if token == BLANK:
                return [1, 0, 0]
            elif token == TOKENS[0]:
                return [0, 1, 0]
            elif token == TOKENS[1]:
                return [0, 0, 1]
            else:
                raise ValueError
        return sum([move_map(token) for token in self.__state], [])

    def clone(self) -> 'Board':
        board = Board(self.num_rows, self.num_cols, self.num_to_win, self.winning_sets)
        board.__state = self.__state
        board.tokens = self.tokens[:]
        board.history = self.history[:]
        board.player_moves = {k: v.copy() for (k, v) in self.player_moves.items()}
        return board

    def make_child_board(self, move: Move) -> 'Board':
        board = self.clone()
        board.make_move(move)
        return board

    def make_move(self, move: Move) -> None:
        self.__state = self.add_token(self.__state, move, self.tokens[0], self.num_cols)
        self.player_moves[self.tokens[0]].add(move)
        self.tokens.append(self.tokens.pop(0))
        self.history.append(move)

    def get_child_boards(self) -> List[Tuple[Move, 'Board']]:
        return [(move, self.make_child_board(move)) for move in self.get_available_moves()]

    def is_full(self) -> bool:
        return all(token != BLANK for token in self.__state)

    def is_move_free(self, move: Move):
        row_idx, col_idx = move
        return self.__state[col_idx + self.num_cols * row_idx] == BLANK

    def get_available_moves(self) -> List[Move]:
        matching_moves = []
        for idx, space in enumerate(self.__state):
            if space == BLANK:
                row_idx = int(idx / self.num_cols)
                col_idx = idx % self.num_cols
                matching_moves.append((row_idx, col_idx))
        return matching_moves

    def get_outcome(self) -> Optional[str]:
        for token in TOKENS:
            if self.check_for_win(self.player_moves[token]):
                return token
        if self.is_full():
            return DRAW
        return UNDETERMINED

    def does_move_win(self, move: Move, token: str) -> bool:
        move_set_copy = self.player_moves[token].copy()
        move_set_copy.add(move)
        return self.check_for_win(move_set_copy)

    def check_for_win(self, move_set: Set[Move]) -> bool:
        for winning_set in self.winning_sets:
            if winning_set.issubset(move_set):
                return True
        return False

    @staticmethod
    def add_token(state: State, move: Move, token: str, num_cols: int) -> State:
        state_clone = state[:]
        row_idx, col_idx = move
        if state[col_idx + num_cols * row_idx] != BLANK:
            raise ValueError('Space taken')
        state_clone[col_idx + num_cols * row_idx] = token
        return state_clone

    def serializable_history(self) -> str:
        return json.dumps([list(move) for move in self.history])


def get_winning_sets(num_rows: int, num_cols: int, num_to_win: int) -> WinningSets:
    horizontal_starts = []
    diagonal_starts = []
    vertical_starts = []
    antidiagonal_starts = []
    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            start = (row_idx, col_idx)
            if row_idx + num_to_win <= num_rows and col_idx + num_to_win <= num_cols:
                diagonal_starts.append(start)
            if row_idx + num_to_win <= num_rows:
                vertical_starts.append(start)
            if col_idx + num_to_win <= num_cols:
                horizontal_starts.append(start)
            if row_idx + num_to_win <= num_rows and col_idx + 1 - num_to_win >= 0:
                antidiagonal_starts.append(start)

    winning_sets = []
    for h_start in horizontal_starts:
        horizontal_set = set((h_start[0], h_start[1] + offset) for offset in range(num_to_win))
        winning_sets.append(horizontal_set)
    for d_start in diagonal_starts:
        diagonal_set = set((d_start[0] + offset, d_start[1] + offset) for offset in range(num_to_win))
        winning_sets.append(diagonal_set)
    for v_start in vertical_starts:
        vertical_set = set((v_start[0] + offset, v_start[1]) for offset in range(num_to_win))
        winning_sets.append(vertical_set)
    for a_start in antidiagonal_starts:
        antidiagonal_set = set((a_start[0] + offset, a_start[1] - offset) for offset in range(num_to_win))
        winning_sets.append(antidiagonal_set)
    return winning_sets
