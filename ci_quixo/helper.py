from typing import TYPE_CHECKING, TypedDict
try:
    from main import Move, RandomPlayer, Game, Player
    from custom_game import CustomGame
    if TYPE_CHECKING:
        from custom_game import CompleteMove
except: 
    from .main import Move, RandomPlayer, Game, Player
    from .custom_game import CustomGame
    if TYPE_CHECKING:
        from .custom_game import CompleteMove

import time
import numpy as np
from tqdm.auto import trange
from os import path
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from pprint import pformat
PLOT_FOLDER = path.join(path.dirname(path.abspath(__file__)), "results")

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

@dataclass
class Result:
    p1: Player
    p2: Player
    wrfs: tuple[float]
    """Player 1 Win Rate (as first, as second)"""
    avg_time: float

    @property
    def wr(self) -> float:
        return sum(self.wrfs)/2


def evaluate(p1: "Player", p2: "Player" = None, games: int = 10, display: bool = False, hide_pbar: bool = False) -> Result:
    if games % 2 != 0:
        games += 1
    if p2 is None:
        p2 = RandomPlayer()
    wins_as_first, wins_as_second = 0, 0
    pbar = trange(games, desc="Evaluating player", unit="game") if not hide_pbar else range(games)
    p1name = "RandomPlayer" if isinstance(p1, RandomPlayer) else p1.name
    p2name = "RandomPlayer" if isinstance(p2, RandomPlayer) else p2.name
    
    durations = []
    for i in pbar:
        game = Game()
        s = time.time()
        if i % 2 == 0:
            wins_as_first  += 1 if game.play(p1, p2) == 0 else 0
        else:
            wins_as_second += 1 if game.play(p2, p1) == 1 else 0
        wins = wins_as_first + wins_as_second
        durations.append(time.time() - s)
        not hide_pbar and pbar.set_postfix({"wins": f"{(wins/(i+1)):.1%}"})
    wins /= games
    wins_as_first /= games/2        
    wins_as_second /= games/2
    avg=sum(durations)/len(durations)
    if display:
        print(f"----- {p1name} vs {p2name} w/ {games} games -----")
        print(f"Total wins  : {wins:>6.2%}")
        print(f"Wins as 1st : {wins_as_first:>6.2%}")
        print(f"Wins as 2nd : {wins_as_second:>6.2%}")
        print(f"Average Time: {avg:>6.2f}s")
        try:
            print(f"{p1.name} stats: \n{pformat(p1.stats)}")
        except:
            pass
        try:
            print(f"{p2.name} stats: \n{pformat(p2.stats)}")
        except:
            pass
        print("----- ------")
    # gen_plot("a", ("a", "a"), [1, 0])
    return Result(p1, p2, (wins_as_first, wins_as_second), avg)


def gen_plots(results: list[Result]) -> None:
    plot_time_comparison(results)
    plot_wr_vs_random(results)
    
    pass

def plot_wr_vs_random(results: list[Result]) -> None:
    results = [r for r in results if isinstance(r.p2, RandomPlayer)]
    
    player_names = [r.p1.short_name for r in results]
    wr1 = [r.wrfs[0] * 100 for r in results]
    wr2 = [r.wrfs[1] * 100 for r in results]
    wr = [r.wr * 100 for r in results]

    values = {
        "Played as 1st": wr1,
        "Played as 2nd": wr2,
        "Overall": wr
    }

    x = np.arange(len(player_names)) * 1.6
    width = 0.30  # the width of the bars

    
    fig, ax = plt.subplots(layout='constrained')
    multiplier = 0
    for attribute, measurement in values.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Agents Win Rate vs Random')
    ax.set_xticks(x + width, player_names)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 120)

    plt.savefig(path.join(PLOT_FOLDER, "players_wr.png"))


def plot_time_comparison(results: list[Result]) -> None:
    # Only vs Random
    results = sorted([r for r in results if isinstance(r.p2, RandomPlayer)], key=lambda it: it.avg_time)
    ys = [r.avg_time for r in results]
    xs = [r.p1.short_name for r in results]
    plt_title = "Average Seconds per Game (vs Random)"
    plt.plot(xs, ys, 'o')
    plt.title(plt_title)
    plt.ylabel("seconds")
    plt.yticks(ys)
    plt.xticks(list(range(len(xs))), xs, rotation=0)
    plt.savefig(path.join(PLOT_FOLDER, "time_comparison.png"))
    



class HumanPlayer(Player):
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
