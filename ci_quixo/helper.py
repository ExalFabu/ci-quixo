import numpy as np
from typing import TYPE_CHECKING
from main import Move, RandomPlayer, Game
from tqdm.auto import trange
from custom_game import CustomGame
from main import Player
if TYPE_CHECKING:
    from custom_game import CompleteMove
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

def evaluate(p1: "Player", p2: "Player" = None, games: int = 10, display: bool = False) -> tuple[int]:
    if games % 2 != 0:
        games += 1
    if p2 is None:
        p2 = RandomPlayer()
    won_as_first, won_as_second = 0, 0
    for i in trange(games, desc="Evaluating player", unit="game"):
        game = Game()
        if i % 2 == 0:
            won_as_first  += 1 if game.play(p1, p2) == 0 else 0
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

class Human(Player):
    def make_move(self, game: "Game") -> "CompleteMove":
        print("---\nIt's your turn")
        CustomGame.from_game(game).pprint()
        x, y, dir = None, None, None
        while (x is None or y is None or dir is None) or not (0 <= x <= 4) or not (0 <= y <= 4) or not dir in ['w', 'a', 's', 'd', 't', 'b', 'l', 'r']:
            try:
                x, y, dir = input("Your move (x, y, dir): ").split(" ")
                if not x.isdigit() or not y.isdigit():
                    x, y = None, None
                    continue
                x, y = int(x), int(y)
            except ValueError:
                continue
        match dir:
            case 'b':
                dir = Move.BOTTOM
            case 's':
                dir = Move.BOTTOM
            case 'u':
                dir = Move.TOP
            case 'w':
                dir = Move.TOP
            case 'a':
                dir = Move.LEFT
            case 'l': 
                dir = Move.LEFT
            case 'r':
                dir = Move.RIGHT
            case 'd':
                dir = Move.RIGHT
        return ((x,y), dir)
