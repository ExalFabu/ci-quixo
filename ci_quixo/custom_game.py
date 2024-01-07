import numpy as np
from game import Game, Move
from collections import namedtuple, defaultdict
from copy import deepcopy

Position = namedtuple("Position", ["x", "y"], defaults=[0, 0])
POSSIBLE_POSITIONS = tuple(
    {Position(0, i) for i in range(5)}
    .union({Position(i, 0) for i in range(5)})
    .union({Position(4, i) for i in range(5)})
    .union({Position(i, 4) for i in range(5)})
)
"""Possible positions (perimeter)"""
CompleteMove = namedtuple("CompleteMove", ["position", "move"])


def valid_move_from_position(p: Position) -> list[Move]:
    valids = []
    if p.x != 0:
        valids.append(Move.LEFT)
    if p.x != 4:
        valids.append(Move.RIGHT)
    if p.y != 0:
        valids.append(Move.TOP)
    if p.y != 4:
        valids.append(Move.BOTTOM)

    return valids


POSSIBLE_MOVES = tuple(
    CompleteMove(p, m) for p in POSSIBLE_POSITIONS for m in valid_move_from_position(p) 
)
"""Every possible moves, taking into account the position in the board (obviously, not considering the board itself)"""

INT_TO_CHAR = ["B", "X", "O"]
"""To stringify board"""

CHARS_TO_INT = {
    "B": -1,
    "X": 0,
    "O": 1
}
"""To parse stringified version back into Game"""


class CustomGame(Game):
    def pprint(self):
        chars = np.ndarray(self._board.shape, np.dtypes.StrDType)
        chars[self._board == -1] = "⬜"
        chars[self._board == 0] = "❎"
        chars[self._board == 1] = "🔵"
        for row in chars:
            for c in row:
                print(c, end="")
            print()

    def __repr__(self) -> str:
        return f"{self.current_player_idx}|{self._board.flatten()}"

    def __str__(self) -> str:
        arr: list[int] = deepcopy(self._board).flatten().tolist()
        stringified = "".join([INT_TO_CHAR[it + 1] for it in arr])
        return f"{self.current_player_idx}{stringified}"
    
    def from_board(board: np.ndarray, player_idx: int) -> "CustomGame":
        c = CustomGame()
        c._board = board
        c.current_player_idx = player_idx
        return c

    def from_str(s: str) -> "CustomGame":
        p, b = s[0], s[1:]
        assert len(b) == 25 and p.isdigit(), f"Invalid Board {s} or playerind {p} ???"
        board  = np.array([CHARS_TO_INT[c] for c in b]).reshape((5,5))
        g = CustomGame()
        g._board = board
        g.current_player_idx = int(p)
        return g
    
    def symmetries(start: "CustomGame") -> list[str]:
        starting_board = start.get_board()
        rotations = [CustomGame.from_board(np.rot90(starting_board, k=k), start.current_player_idx) for k in range(4)]
        flip = np.fliplr(starting_board)
        flip_rotations = [CustomGame.from_board(np.rot90(flip, k=k), start.current_player_idx) for k in range(4)] 
        inverted = deepcopy(starting_board)
        zeros = inverted == 0
        ones = inverted == 1
        inverted[zeros] = 1
        inverted[ones] = 0
        inv_rotations = [CustomGame.from_board(np.rot90(inverted, k=k), 1-start.current_player_idx) for k in range(4)]
        all_variants = set([*rotations, *flip_rotations, *inv_rotations])
        return sorted([str(it) for it in list(all_variants)])
    
    def to_canon(start: "CustomGame") -> tuple["CustomGame", int]:
        symmetries = start.symmetries()
        self_idx = symmetries.index(str(start))
        return CustomGame.from_str(symmetries[0]), self_idx
    
    def from_canon(canon: "CustomGame", idx: int) -> "CustomGame":
        symmetries = canon.symmetries()
        return CustomGame.from_str(symmetries[idx])
    
    def from_game(game: "Game") -> "CustomGame":
        return CustomGame.from_board(game.get_board(), game.get_current_player())

    def __hash__(self) -> str:
        return str(self).__hash__()

    def __eq__(self, other: "CustomGame") -> bool:
        return self.__hash__() == other.__hash__()
    
    def valid_moves(self, player: int, filter_duplicates: bool = True) -> tuple[CompleteMove]:
        valids = [it for it in POSSIBLE_MOVES if self._board[it.position[::-1]] == -1 or self._board[it.position[::-1]] == player]
        if not filter_duplicates:
            return valids
        s = defaultdict(list)
        for valid in valids:
            copy = deepcopy(self)
            copy._Game__move(*valid, player)
            s[str(copy)].append(valid)
        non_duplicate = []
        for _, moves in s.items():
            non_duplicate.append(moves[0])
        return tuple(non_duplicate)
    def score(self) -> int:
        winner = self.check_winner()
        if winner != -1:
            return (5**5) * 1 if winner == self.current_player_idx else -1
        transposed = self._board.transpose()
        
        x_score = []
        o_score = []
        for row, column in zip(self._board, transposed):
            x_score.append(sum(row == 0))
            x_score.append(sum(column == 0))
            o_score.append(sum(row == 1))
            o_score.append(sum(column == 1))
        
        diag = self._board.diagonal()
        second_diag = self._board[:, ::-1].diagonal()

        x_score.append(sum(diag == 0))
        o_score.append(sum(diag == 1))
        x_score.append(sum(second_diag == 0))
        o_score.append(sum(second_diag == 1))

        score_x, score_o = 5**max(x_score), 5**max(o_score)
        score = score_x - score_o
        score *= 1 if self.current_player_idx == 0 else -1
        return score

if __name__ == "__main__":
    from random import choice
    a = CustomGame()
    p = CompleteMove((0, 0), Move.BOTTOM)
    a._Game__move(*choice(POSSIBLE_MOVES), choice([0, 1]))
