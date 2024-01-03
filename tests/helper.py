# Basic test template
from ci_quixo.game import Game
from ci_quixo.helper import pprint_board


def test_print():
    g = Game()
    pprint_board(g)