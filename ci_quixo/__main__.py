try:
    from _players import MinMaxPlayer, MCTSPlayer, RandomPlayer
    from helper import evaluate, Result, gen_plots
    from main import Player
except:
    from ._players import *
    from .main import Player
    from .helper import evaluate, Result, gen_plots
from pprint import pprint
import dill, multiprocessing
dill.Pickler.dumps, dill.Pickler.loads = dill.dumps, dill.loads
multiprocessing.reduction.ForkingPickler = dill.Pickler
multiprocessing.reduction.dump = dill.dump

GAMES = 10
HIDE_PBAR = True

minmax2 = MinMaxPlayer()
minmax3 = MinMaxPlayer(3, pruning=2)
mcts_r = MCTSPlayer()
mcts_h = MCTSPlayer(sim_heuristic=True)


def test_minmax_vs_random() -> Result:
    return evaluate(minmax2, RandomPlayer(), games=GAMES, display=True, hide_pbar=HIDE_PBAR)

def test_minmax3_vs_random() -> Result:
    return evaluate(minmax3, RandomPlayer(), games=GAMES, display=True, hide_pbar=HIDE_PBAR)

def test_mcts_r_vs_random() -> Result:
    return evaluate(mcts_r, RandomPlayer(), games=GAMES, display=True, hide_pbar=HIDE_PBAR)


def test_mcts_h_vs_random() -> Result:
    return evaluate(mcts_h, RandomPlayer(), games=GAMES, display=True, hide_pbar=HIDE_PBAR)


def test_minmax_vs_mcts_random() -> Result:
    return evaluate(minmax2, mcts_r, games=GAMES, display=True, hide_pbar=HIDE_PBAR)

def test_minmax_vs_mcts_heuristic() -> Result:
    return evaluate(minmax2, mcts_h, games=GAMES, display=True, hide_pbar=HIDE_PBAR)

def call(it: callable) -> Result:
    return it()

if __name__ == "__main__":
    evals = [test_minmax_vs_random, test_minmax3_vs_random, test_mcts_r_vs_random, test_mcts_h_vs_random, test_minmax_vs_mcts_random, test_minmax_vs_mcts_heuristic]

    with multiprocessing.Pool() as p:
        RESULTS = p.map(call, evals[:-2])
        
    gen_plots(RESULTS)