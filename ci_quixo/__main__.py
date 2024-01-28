try:
    from _players import MinMaxPlayer, MCTSPlayer, RandomPlayer
    from helper import evaluate, Result, gen_plots
    from main import Player
except:
    from ._players import *
    from .main import Player
    from .helper import evaluate, Result, gen_plots
import dill, multiprocessing
from typing import Callable
dill.Pickler.dumps, dill.Pickler.loads = dill.dumps, dill.loads
multiprocessing.reduction.ForkingPickler = dill.Pickler
multiprocessing.reduction.dump = dill.dump

GAMES = 100
HIDE_PBAR = True
PARALLELIZE = True
DISPLAY_RES = True

minmax2 = MinMaxPlayer()
minmax3 = MinMaxPlayer(3, pruning=2)
mcts_r = MCTSPlayer()
mcts_h = MCTSPlayer(sim_heuristic=True)


def test_minmax_vs_random() -> Result:
    return evaluate(minmax2, RandomPlayer(), games=GAMES, display=DISPLAY_RES, hide_pbar=HIDE_PBAR)

def test_minmax3_vs_random() -> Result:
    return evaluate(minmax3, RandomPlayer(), games=GAMES, display=DISPLAY_RES, hide_pbar=HIDE_PBAR)

def test_mcts_r_vs_random() -> Result:
    return evaluate(mcts_r, RandomPlayer(), games=GAMES, display=DISPLAY_RES, hide_pbar=HIDE_PBAR)


def test_mcts_h_vs_random() -> Result:
    return evaluate(mcts_h, RandomPlayer(), games=GAMES, display=DISPLAY_RES, hide_pbar=HIDE_PBAR)


def test_minmax_vs_mcts_random() -> Result:
    return evaluate(minmax2, mcts_r, games=GAMES, display=DISPLAY_RES, hide_pbar=HIDE_PBAR)

def test_minmax_vs_mcts_heuristic() -> Result:
    return evaluate(minmax2, mcts_h, games=GAMES, display=DISPLAY_RES, hide_pbar=HIDE_PBAR)

def call(it: Callable[[], Result]) -> Result:
    return it()

def main(tests: list[Callable[[], Result]]): 
    from tqdm import tqdm

    if len(tests) == 0:
        gen_plots(tests)
        return

    if PARALLELIZE:
        with multiprocessing.Pool() as p:
            RESULTS = []
            pbar = tqdm(p.imap(call, tests), total=len(tests), unit="test", desc=f"Testing with {GAMES} games each")
            for it in pbar:
                RESULTS.append(it)

    else:
        RESULTS = []
        pbar = tqdm(tests, unit="test", desc=f"Testing with {GAMES} games each")
        for it in pbar:
            pbar.set_postfix({"current": it.__name__})
            res = it()
            RESULTS.append(res)
    gen_plots(RESULTS)


if __name__ == "__main__":
    VS_RANDOM = [test_minmax_vs_random, test_minmax3_vs_random, test_mcts_r_vs_random, test_mcts_h_vs_random]
    VS_EACHOTHER = [test_minmax_vs_mcts_random, test_minmax_vs_mcts_heuristic]
    ALL_ = [*VS_RANDOM, *VS_EACHOTHER]
    # To use saved results
    main([])
    # To recalculate all of them
    # main(ALL_)
    # To test only one of them
    # main([test_minmax_vs_random])

    # To run this file, python -m ci_quixo (while in folder ci_quixo, not inside ci_quixo/ci_quixo)

