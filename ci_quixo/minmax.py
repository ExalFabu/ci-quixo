from game import Player, Game
from custom_game import CustomGame
from typing import TYPE_CHECKING
import numpy as np
from copy import deepcopy
import random

if TYPE_CHECKING:
    from custom_game import CompleteMove

class MinMaxPlayer(Player):
    
    def __init__(self, max_depth: int = None, use_alpha_beta_pruning: bool = False, verbose: bool = False) -> None:
        super().__init__()

        self.max_depth = max_depth
        self.use_alpha_beta_pruning = use_alpha_beta_pruning
        self.verbose = verbose

    def make_move(self, game: Game) -> "CompleteMove":
        self.verbose and print("Deciding move on the following board")
        cg = CustomGame.from_game(game)
        self.verbose and cg.pprint()
        best_move = self._minmax(cg)
        if best_move is None:
            best_move = random.choice(cg.valid_moves(cg.get_current_player()))
        cg._Game__move(*best_move, cg.current_player_idx)
        self.verbose and print(f"Played {best_move=}")
        self.verbose and cg.pprint()
        return best_move
    
    def _minmax(self, game: "CustomGame") -> "CompleteMove":

        def min_side(self: "MinMaxPlayer", game: "CustomGame", alpha: int, beta: int, depth: int) -> int:
            winner = game.check_winner()
            if (self.max_depth is not None and depth >= self.max_depth) or winner != -1:
                return game.score()
        
            min_found = np.infty

            for move in game.valid_moves(game.current_player_idx):
                copy = deepcopy(game)
                assert copy._Game__move(*move, copy.current_player_idx), "Somehow move was invalid?????"
                copy.current_player_idx = 1-copy.current_player_idx
                min_found = min(min_found, max_side(self, game, alpha, beta, depth+1))
                beta = min(beta, min_found)
                if alpha > beta and self.use_alpha_beta_pruning:
                    break
            return min_found

            
        def max_side(self: "MinMaxPlayer", game: "CustomGame", alpha: int, beta: int, depth: int) -> int:
            winner = game.check_winner()
            if (self.max_depth is not None and depth >= self.max_depth) or winner != -1:
                return game.score()
                
            max_found = -np.infty
            
            for move in game.valid_moves(game.current_player_idx):
                copy = deepcopy(game)
                assert copy._Game__move(*move, copy.current_player_idx), "Somehow move was invalid?????"
                copy.current_player_idx = 1-copy.current_player_idx
                max_found = max(max_found, min_side(self, game, alpha, beta, depth+1))
                alpha = max(alpha, max_found)
                if alpha > beta and self.use_alpha_beta_pruning:
                    break
            return max_found

        best_move = None
        alpha, beta = -np.inf, np.inf

        for move in game.valid_moves(game.current_player_idx):
            copy = deepcopy(game)
            assert copy._Game__move(*move, copy.current_player_idx), "Somehow move was invalid?????"
            copy.current_player_idx = 1-copy.current_player_idx
            min_score = min_side(self, game, alpha, beta, 1)
            if min_score > alpha:
                alpha = min_score
                best_move = move
        self.verbose and print(f"Found best move with score {alpha}")
        return best_move


if __name__ == "__main__":
    from helper import evaluate
    evaluate(MinMaxPlayer(3, True), None, 10, True)