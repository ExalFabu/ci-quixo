from game import Player, Game
from custom_game import CustomGame, POSSIBLE_MOVES
from typing import TYPE_CHECKING
import numpy as np
from collections import defaultdict
import random

if TYPE_CHECKING:
    from custom_game import CompleteMove

class MinMaxPlayer(Player):
    
    def __init__(self, max_depth: int = None, *, alpha_beta: bool = True, pruning: bool = True, verbose: bool = False) -> None:
        super().__init__()

        self.max_depth = 2 if max_depth is None else max_depth
        self.use_alpha_beta_pruning = alpha_beta
        self.use_symmetries = pruning
        self.verbose = verbose
        self.history: dict[str, "CompleteMove"] = dict()
        self.stats = defaultdict(int)

    def make_move(self, game: Game) -> "CompleteMove":
        cg = CustomGame.from_game(game)
        best_move = self._minmax(cg)
        if best_move is None or not cg.is_valid(best_move):
            self.stats['EVAL-invalidmove'] += 1
            best_move = random.choice(cg.valid_moves())
        return best_move
    
    def _minmax(self, game: "CustomGame") -> "CompleteMove":

        def moves_getter(game: "CustomGame") -> list["CompleteMove"]:
            if self.use_symmetries is None:
                return POSSIBLE_MOVES
            elif self.use_symmetries is True:
                return game.valid_moves(None, True, True)
            else: 
                return game.valid_moves(None, True, True)

        def min_side(self: "MinMaxPlayer", game: "CustomGame", alpha: int, beta: int, depth: int) -> int:
            winner = game.check_winner()
            if (self.max_depth is not None and depth >= self.max_depth) or winner != -1:
                self.stats[f"m-w{winner}-{depth}"] += 1
                return game.score
        
            min_found = np.infty

            for move in moves_getter(game):
                copy = game.simulate_move(move)
                copy.current_player_idx = 1-copy.current_player_idx
                min_found = min(min_found, max_side(self, copy, alpha, beta, depth+1))
                beta = min(beta, min_found)
                if alpha > beta and self.use_alpha_beta_pruning:
                    break
            return min_found

            
        def max_side(self: "MinMaxPlayer", game: "CustomGame", alpha: int, beta: int, depth: int) -> int:
            winner = game.check_winner()
            if (self.max_depth is not None and depth >= self.max_depth) or winner != -1:
                self.stats[f"M-w{winner}-{depth}"] += 1
                return game.score
                
            max_found = -np.infty
            
            for move in moves_getter(game):
                copy = game.simulate_move(move)
                max_found = max(max_found, min_side(self, copy, alpha, beta, depth+1))
                alpha = max(alpha, max_found)
                if alpha > beta and self.use_alpha_beta_pruning:
                    break
            return max_found

        best_move = None
        alpha, beta = -np.inf, np.inf

        if str(game) in self.history:
            self.stats['cache-hit'] += 1
            return self.history[str(game)]

        for move in moves_getter(game):
            copy = game.simulate_move(move)
            min_score = min_side(self, copy, alpha, beta, 1)
            if min_score > alpha:
                alpha = min_score
                best_move = move
        self.stats['EVALS'] += 1
        self.history[str(game)] = best_move
        return best_move


if __name__ == "__main__":
    from helper import evaluate
    m = MinMaxPlayer(pruning=True)
    evaluate(m, None, 10, True)