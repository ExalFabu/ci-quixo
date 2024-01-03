import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from game import Game

def pprint_board(game: "Game"):
    board: np.ndarray = game.get_board()
    chars = np.ndarray(board.shape, np.dtypes.StrDType)
    chars[board ==-1] = '⬜'
    chars[board == 0] = '❎'
    chars[board == 1] = '🔵'
    for row in chars:
        for c in row:
            print(c, end="")
        print()

