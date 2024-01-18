from ._players import *
from .custom_game import CustomGame, Game
from .helper import evaluate
import pytest

@pytest.mark.evaluate
def test_minmax_vs_random():
    minmax = MinMaxPlayer()

    print("Minmax VS RandomPlayer")
    evaluate(minmax, RandomPlayer(), games=100, display=True)

@pytest.mark.evaluate
def test_mcts_r_vs_random():
    mcts = MCTSPlayer()

    print("MCTS(Random) vs RandomPlayer")
    evaluate(mcts, RandomPlayer(), games=100, display=True)

@pytest.mark.evaluate
def test_mcts_h_vs_random():
    mcts = MCTSPlayer(sim_heuristic=True)

    print("MCTS(Random) vs RandomPlayer")
    evaluate(mcts, RandomPlayer(), games=100, display=True)

@pytest.mark.evaluate
def test_minmax_vs_mcts_random():
    minmax = MinMaxPlayer()
    mcts = MCTSPlayer()

    print("Minmax VS MCTS(Random)")
    evaluate(minmax, mcts, games=100, display=True)

@pytest.mark.evaluate
def test_minmax_vs_mcts_heuristic():
    minmax = MinMaxPlayer()
    mcts = MCTSPlayer(sim_heuristic=True)

    print("Minmax VS MCTS(Heuristic)")
    evaluate(minmax, mcts, games=100, display=True)

if __name__ == "__main__":
    print("Use `python -m pytest ci_quixo/__main__.py -m evaluate -s`")