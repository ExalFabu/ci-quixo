try:
    from _players import MinMaxPlayer, MCTSPlayer, RandomPlayer
    from helper import evaluate
except:
    from ._players import *
    from .helper import evaluate
import pytest
from pprint import pprint

GAMES = 10
HIDE_PBAR = True

@pytest.mark.evaluate
def test_minmax_vs_random():
    minmax = MinMaxPlayer()

    print("\n--- --- ---")
    print(f"{minmax.name} VS RandomPlayer")
    evaluate(minmax, RandomPlayer(), games=GAMES, display=True, hide_pbar=HIDE_PBAR)
    print(f"{minmax.name} Stats:")
    pprint(minmax.stats, sort_dicts=False)

@pytest.mark.evaluate
def test_minmax3_vs_random():
    minmax = MinMaxPlayer(3, pruning=2)

    print("\n--- --- ---")
    print(f"{minmax.name} VS RandomPlayer")
    evaluate(minmax, RandomPlayer(), games=GAMES, display=True, hide_pbar=HIDE_PBAR)
    print(f"{minmax.name} Stats:")
    pprint(minmax.stats, sort_dicts=False)

@pytest.mark.evaluate
def test_mcts_r_vs_random():
    mcts = MCTSPlayer()

    print("\n--- --- ---")
    print(f"{mcts.name} vs RandomPlayer")
    evaluate(mcts, RandomPlayer(), games=GAMES, display=True, hide_pbar=HIDE_PBAR)
    print(f"{mcts.name} Stats:")
    pprint(mcts.stats, sort_dicts=False)


@pytest.mark.evaluate
def test_mcts_h_vs_random():
    mcts = MCTSPlayer(sim_heuristic=True)

    print("\n--- --- ---")
    print(f"{mcts.name} vs RandomPlayer")
    evaluate(mcts, RandomPlayer(), games=GAMES, display=True, hide_pbar=HIDE_PBAR)
    print(f"{mcts.name} Stats:")
    pprint(mcts.stats, sort_dicts=False)


@pytest.mark.evaluate
def test_minmax_vs_mcts_random():
    minmax = MinMaxPlayer()
    mcts = MCTSPlayer()
    print("\n--- --- ---")
    print(f"{minmax.name} VS {mcts.name}")
    evaluate(minmax, mcts, games=GAMES, display=True, hide_pbar=HIDE_PBAR)
    print(f"{minmax.name} Stats:")
    pprint(minmax.stats, sort_dicts=False)
    print(f"{mcts.name} Stats:")
    pprint(mcts.stats, sort_dicts=False)

@pytest.mark.evaluate
def test_minmax_vs_mcts_heuristic():
    minmax = MinMaxPlayer()
    mcts = MCTSPlayer(sim_heuristic=True)

    print("\n--- --- ---")
    print(f"{minmax.name} VS {mcts.name}")
    evaluate(minmax, mcts, games=GAMES, display=True, hide_pbar=HIDE_PBAR)
    print(f"{minmax.name} Stats:")
    pprint(minmax.stats, sort_dicts=False)
    print(f"{mcts.name} Stats:")
    pprint(mcts.stats, sort_dicts=False)


if __name__ == "__main__":
    print("Use `python -m pytest ci_quixo/__main__.py -m evaluate -s`")