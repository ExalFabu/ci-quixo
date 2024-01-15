import numpy as np
from game import Game, Move
from collections import namedtuple, defaultdict
from copy import deepcopy
from typing import TYPE_CHECKING
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from main import Player

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
    
    @property
    def next_move_for(self) -> int:
        """This is a fix to the fact that current_player_idx somehow indicates the player that has already moved (1 on game start??)"""
        return 1-self.current_player_idx

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
        def rot_flip(board: np.ndarray, player_idx: int) -> list["CustomGame"]:
            starting_board = board
            rotations = [CustomGame.from_board(np.rot90(starting_board, k=k), player_idx) for k in range(4)]
            flip = np.fliplr(starting_board)
            flip_rotations = [CustomGame.from_board(np.rot90(flip, k=k), player_idx) for k in range(4)]
            return [*rotations, *flip_rotations]
        
        inverted = start.get_board()
        zeros = inverted == 0
        ones = inverted == 1
        inverted[zeros] = 1
        inverted[ones] = 0
        
        all_variants = set([*rot_flip(start.get_board(), start.current_player_idx), *rot_flip(inverted, 1-start.current_player_idx)])
        # all_variants = set([*rot_flip(start.get_board(), start.current_player_idx)])
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
    
    @staticmethod
    def convert_canon_move(canon_board: "CustomGame", canon_move: "CompleteMove", original_board: "CustomGame") -> "CompleteMove":
        target_board = str(canon_board.simulate_move(canon_move).to_canon()[0])
        for move in original_board.valid_moves(None, False):
            temp_board = original_board.simulate_move(move)
            if str(temp_board.to_canon()[0]) == target_board:
                return move
        debug = f"canon= {canon_board} move= {canon_move} original= {original_board}"
        raise Exception(f"Unable to convert move from canon to non-canon\n{debug}")
    
    def valid_moves(self, player: int = None, filter_duplicates: bool = True, canon_unique: bool = False) -> tuple[CompleteMove]:
        if player is None:
            player = self.current_player_idx
        valids = [it for it in POSSIBLE_MOVES if self._board[it.position[::-1]] == -1 or self._board[it.position[::-1]] == player]
        if not filter_duplicates:
            return valids
        s = defaultdict(list)
        for valid in valids:
            copy = deepcopy(self)
            copy._Game__move(*valid, player)
            if canon_unique:
                s[str(copy.to_canon()[0])].append(valid)
            else:
                s[str(copy)].append(valid)
        non_duplicate = []
        for _, moves in s.items():
            non_duplicate.append(moves[0])
        return tuple(non_duplicate)
    
    def is_valid(self: "CustomGame", move: "CompleteMove") -> bool:
        return move in self.valid_moves(None, False)
    
    def play(self, player1: "Player", player2: "Player", verbose: bool = False) -> int:
        '''Play the game. Returns the winning player'''
        players = [player1, player2]
        winner = -1
        if verbose:
            pbar = tqdm(range(100))
            pbar.disable = not verbose
            pbar.unit = "move"
        while winner < 0:
            ok = False
            counter = 0
            verbose and pbar.set_postfix({"Player": self.current_player_idx, "wrong-moves": counter})
            while not ok:
                move = players[self.current_player_idx].make_move(self)
                ok = self._Game__move(*move, self.current_player_idx)
                counter += 1
                if verbose and counter > 1:
                    pbar.set_postfix({"Player": self.current_player_idx, "wrong-moves": counter})
            winner = self.check_winner()
            self.current_player_idx = 1-self.current_player_idx
            verbose and pbar.update(1)
        return winner
    
    @property
    def score(self) -> int:
        winner = self.check_winner()
        if winner != -1:
            return (5**5) * (1 if winner == self.current_player_idx else -1)
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
    
    def simulate_move(self, move: "CompleteMove") -> "CustomGame":
        copy = deepcopy(self)
        investigating = copy.is_valid(move)
        success = copy._Game__move(*move, copy.current_player_idx)
        if success:
            copy.current_player_idx = 1-copy.current_player_idx
        assert success == investigating, "AAAA SOMEHOW IS_VALID is different thant Game.move validation | board {copy} - move {move} move for {copy.current_player_idx}"
        return copy

if __name__ == "__main__":
    from random import choice
    a = CustomGame()
    p = CompleteMove((0, 0), Move.BOTTOM)
