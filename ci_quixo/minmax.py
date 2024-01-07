from game import Player, Game
from custom_game import CustomGame
from typing import TYPE_CHECKING
import numpy as np
from copy import deepcopy
import random

if TYPE_CHECKING:
    from custom_game import CompleteMove

class MinMaxPlayer(Player):
    
    def __init__(self, max_depth: int = None, use_alpha_beta_pruning: bool = False) -> None:
        super().__init__()

        self.max_depth = max_depth
        self.use_alpha_beta_pruning = use_alpha_beta_pruning
        self._init_ab()

    def _init_ab(self):
        self._alpha, self._beta = -np.inf, np.inf


    def make_move(self, game: Game) -> "CompleteMove":
        cg = CustomGame.from_game(game)
        best_move = self._minmax(0, cg, True)[1]
        if best_move is None:
            best_move = random.choice(cg.valid_moves())
        print("I made a move...")
        return best_move
    
    def _minmax(self, depth: int, game: "CustomGame", maximixe: bool) -> tuple[float, "CompleteMove"]:
        winner = game.check_winner()
        if winner != -1:
            return 25 * winner, None
        if self.max_depth is not None and depth >= self.max_depth:
            return 0, None

        if depth == 0:
            self._init_ab()        

        best_move = None
        if maximixe:
            for move in game.valid_moves(game.get_current_player()):
                copied = deepcopy(game)
                assert copied._Game__move(*move, copied.current_player_idx), f"Somehow got an invalid move while iterating from valid moves, {copied}, {move}"
                score, _ = self._minmax(depth+1, copied, False)
                if score > self._alpha:
                    self._alpha = score
                    best_move = move
                
                if self.use_alpha_beta_pruning and self._alpha > self._beta:
                    break
            return self._alpha, best_move
        else:
            for move in game.valid_moves(game.get_current_player()):
                copied = deepcopy(game)
                assert copied._Game__move(*move, copied.current_player_idx), "Somehow got an invalid move while iterating from valid moves"
                score, _ = self._minmax(depth+1, copied, True)
                if score < self._beta:
                    self._beta = score
                    best_move = move
                
                if self.use_alpha_beta_pruning and self._alpha > self._beta:
                    break
            return self._beta, best_move

if __name__ == "__main__":
    from main import RandomPlayer
    mm = MinMaxPlayer(20, True)
    rp = RandomPlayer()
    game = Game()