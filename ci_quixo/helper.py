import numpy as np
from typing import TYPE_CHECKING
from main import RandomPlayer, Game
from tqdm.auto import trange
from custom_game import CustomGame
if TYPE_CHECKING:
    from game import Game
    from main import Player

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

def evaluate(p1: "Player", p2: "Player" = None, games: int = 10, display: bool = False) -> tuple[int]:
    if p2 is None:
        p2 = RandomPlayer()
    won_as_first, won_as_second = 0, 0
    for i in trange(games, desc="Evaluating player", unit="game"):
        game = CustomGame()
        if i % 2 == 0:
            won_as_first += 1 if game.play(p1, p2) == 0 else 0
        else:
            won_as_second += 1 if game.play(p2, p1) == 1 else 0
    wins = won_as_first + won_as_second
    wins /= games
    won_as_first /= games/2        
    won_as_second /= games/2
    if display:
        print(f"Total wins : {wins:.2%}")
        print(f"Wins as 1st: {won_as_first:.2%}")
        print(f"Wins as 2nd: {won_as_second:.2%}")
    return wins, won_as_first, won_as_second