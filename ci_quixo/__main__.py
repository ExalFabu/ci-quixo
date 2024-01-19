try:
    from _players import *
    from helper import evaluate
except:
    from ._players import *
    from .helper import evaluate
import pytest
from pprint import pprint

GAMES = 100

@pytest.mark.evaluate
def test_minmax_vs_random():
    minmax = MinMaxPlayer()

    print("\n--- --- ---")
    print("Minmax VS RandomPlayer")
    evaluate(minmax, RandomPlayer(), games=GAMES, display=True, hide_pbar=True)
    print("MinMax Stats:")
    pprint(minmax.stats, sort_dicts=False)


@pytest.mark.evaluate
def test_mcts_r_vs_random():
    mcts = MCTSPlayer()

    print("\n--- --- ---")
    print("MCTS(Random) vs RandomPlayer")
    evaluate(mcts, RandomPlayer(), games=GAMES, display=True, hide_pbar=True)
    print("MCTS(Random) Stats:")
    pprint(mcts.stats, sort_dicts=False)


@pytest.mark.evaluate
def test_mcts_h_vs_random():
    mcts = MCTSPlayer(sim_heuristic=True)

    print("\n--- --- ---")
    print("MCTS(Heuristic) vs RandomPlayer")
    evaluate(mcts, RandomPlayer(), games=GAMES, display=True, hide_pbar=True)
    print("MCTS(Heuristic) Stats:")
    pprint(mcts.stats, sort_dicts=False)


@pytest.mark.evaluate
def test_minmax_vs_mcts_random():
    minmax: "MinMaxPlayer" = MinMaxPlayer()
    mcts = MCTSPlayer()
    print("\n--- --- ---")
    print("Minmax VS MCTS(Random)")
    evaluate(minmax, mcts, games=GAMES, display=True, hide_pbar=True)
    print("MinMax Stats:")
    pprint(minmax.stats, sort_dicts=False)
    print("MCTS(Random) Stats:")
    pprint(mcts.stats, sort_dicts=False)

@pytest.mark.evaluate
def test_minmax_vs_mcts_heuristic():
    minmax = MinMaxPlayer()
    mcts = MCTSPlayer(sim_heuristic=True)

    print("\n--- --- ---")
    print("Minmax VS MCTS(Heuristic)")
    evaluate(minmax, mcts, games=GAMES, display=True, hide_pbar=True)
    print("MinMax Stats:")
    pprint(minmax.stats, sort_dicts=False)
    print("MCTS(Heuristic) Stats:")
    pprint(mcts.stats, sort_dicts=False)


if __name__ == "__main__":
    print("Use `python -m pytest ci_quixo/__main__.py -m evaluate -s`")