from typing import Optional, Iterable, Dict

from board import Board, Move, DRAW, UNDETERMINED


class DFSGameTreeNode:

    def __init__(self, board: Board, parents: Optional[Dict[Move, 'DFSGameTreeNode']] = None,
                 transpositions: Optional[Dict[int, 'DFSGameTreeNode']] = None, ab_prune: bool = True):
        self.board: Board = board
        self.parents: Dict[Move, DFSGameTreeNode] = parents
        self.transpositions = transpositions or {}
        self.transpositions[hash(self.board)] = self
        self.children: Dict[Move, DFSGameTreeNode] = {}
        self.heuristic: Optional[float] = None
        self.height = None
        self.counts: Dict[str, int] = {self.board.tokens[0]: 0, self.board.tokens[1]: 0, DRAW: 0, UNDETERMINED: 0}
        self.outcome: str = self.board.get_outcome()
        self.get_heuristic()
        self.ab_prune = ab_prune

    def __str__(self):
        return str(self.board) + f"\n outcome: {self.outcome}" + f"\n heuristic: {self.heuristic}"

    def __dict__(self):
        return {'board': str(self.board), 'outcome': self.outcome}

    def get_height(self) -> int:
        return max([child.get_height() for child in self.children.values()] + [0]) + 1

    def get_volume(self) -> int:
        return sum([child.get_volume() for child in self.children.values()], 0) + 1

    def get_children(self) -> Iterable['DFSGameTreeNode']:
        child_boards = [self.board.make_child_board(move) for move in self.board.get_available_moves()]
        child_nodes = []
        for board in child_boards:
            move = board.history[-1]
            if hash(board) in self.transpositions:
                child_node = self.transpositions[hash(board)]
                child_node.parents[move] = self
            else:
                child_node = DFSGameTreeNode(board, transpositions=self.transpositions, parents={move: self},
                                             ab_prune=self.ab_prune)
            child_nodes.append(child_node)
            self.children[move] = child_node
        return child_nodes

    def get_outcome(self):
        if self.outcome is UNDETERMINED:
            child_outcomes = [child.outcome for child in self.children.values()]
            token, opponent_token = self.board.tokens
            if opponent_token in child_outcomes:
                self.outcome = opponent_token
            elif DRAW in child_outcomes:
                self.outcome = DRAW
            elif token in child_outcomes:
                self.outcome = token

    def get_heuristic(self):
        if not self.children:
            token, opponent_token = self.board.tokens
            if self.outcome == DRAW or self.outcome == UNDETERMINED:
                self.heuristic = 0.
            elif self.outcome == token:
                self.heuristic = -1.
            elif self.outcome == opponent_token:
                self.heuristic = 1.
        else:
            self.heuristic = -sum([child.heuristic for child in self.children.values()]) / len(self.children.values())

    def depth_limited_build_tree(self, depth: Optional[int] = None) -> None:
        if depth == 0:
            return
        elif depth is not None:
            depth = depth - 1

        if self.children:
            if self.ab_prune:
                best_heuristic = max([child.heuristic for child in self.children.values()])
            for child in self.children.values():
                if not self.ab_prune or child.heuristic == best_heuristic:
                    child.depth_limited_build_tree(depth)
        else:
            for child in self.get_children():
                if child.outcome is UNDETERMINED:
                    child.depth_limited_build_tree(depth)
        self.get_outcome()
        self.get_heuristic()
        self.height = self.get_height()

    def build_tree(self, max_depth: Optional[int] = None):
        if max_depth is None:
            max_depth = 2 * self.board.num_to_win
        depth = 1
        while self.outcome is UNDETERMINED or depth < max_depth:
            self.depth_limited_build_tree(depth)
            depth += 1
