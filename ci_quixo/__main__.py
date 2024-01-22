try:
    from _players import MinMaxPlayer, MCTSPlayer, RandomPlayer
    from helper import evaluate, Result, gen_plots
    from main import Player
except:
    from ._players import *
    from .main import Player
    from .helper import evaluate, Result, gen_plots
import dill, multiprocessing
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

def call(it: callable) -> Result:
    return it()

if __name__ == "__main__":
    from tqdm import tqdm
    evals = [test_minmax_vs_random, test_minmax3_vs_random, test_mcts_r_vs_random, test_mcts_h_vs_random, test_minmax_vs_mcts_random, test_minmax_vs_mcts_heuristic][:]
    if PARALLELIZE:
        with multiprocessing.Pool() as p:
            RESULTS = list(tqdm(p.imap(call, evals), total=len(evals), unit="test", desc=f"Testing with {GAMES} games each"))
    else:
        RES = [it() for it in tqdm(evals, desc="Evaluating", unit="eval")]        
    gen_plots(RESULTS)