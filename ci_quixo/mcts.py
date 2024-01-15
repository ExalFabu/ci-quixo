from game import Game, Move, Player
from custom_game import CustomGame
import numpy as np, random
from dataclasses import dataclass, field
from collections import defaultdict
from copy import deepcopy
from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from custom_game import CompleteMove

# implementation inspired from https://github.com/aimacode/aima-python/blob/61d695b37c6895902081da1f37baf645b0d2658a/games4e.py#L178

@dataclass
class MCTNode:
    
    state: "CustomGame" = field()
    parent: "MCTNode" = field()
    constant_factor: float = field(default=1.4, init=False)
    utility: int = field(default=0, init=False)
    count: int = field(default=0, init=False)
    children: dict["MCTNode", "CompleteMove"] = field(default_factory=lambda: dict(), init=False)

    def ucb(self, constant_factor = None):
        if self.count == 0:
            return float("inf")
        if constant_factor is None:
            constant_factor = self.constant_factor
        return self.utility / self.count + constant_factor * (np.sqrt(np.log(self.parent.count) / self.count))
    
    def is_terminal(self) -> bool:
        return self.state.check_winner() != -1
    
    def __hash__(self) -> int:
        return hash(str(self.state) + str(hash(self.parent)) + f"{self.utility}/{self.count}")

@dataclass
class MCTSPlayer(Player):
    
    games: int = field(default=1000)
    stats: dict[str, int] = field(default_factory=lambda: defaultdict(int), init=False)
    
    def make_move(self, game: Game) -> tuple[tuple[int, int], Move]:
        
        root_cg = CustomGame.from_game(game)

        root = MCTNode(root_cg, None)

        for _ in range(self.games):
            leaf = self._select(root)
            child = self._expand(leaf)
            score = self._simulate(child)
            self._backpropagate(child, score)
        
        best_move = max(root.children.items(), key=lambda it: it[0].count)[1]
        if best_move not in root_cg.valid_moves(None, False, False):
            self.stats['eval-invalid'] += 1
        return best_move

    def _select(self, node: "MCTNode") -> "MCTNode":
        if node.children:
            return self._select(max(node.children.keys(), key=MCTNode.ucb))
        else:
            return node
        
    def _expand(self, node: "MCTNode") -> "MCTNode":
        if not node.children or not node.is_terminal():
            node.children = {
                MCTNode(node.state.simulate_move(move), node): move
                for move in node.state.valid_moves(None, True, True)
            }
        return self._select(node)
    
    def _select_move_in_simulation(self, game: "CustomGame") -> "CompleteMove":
        return random.choice(game.valid_moves(None, True, True))
        
    def _simulate(self, node: "MCTNode") -> int:
        starting_player = node.state.get_current_player()
        copy = deepcopy(node.state)
        winner = copy.check_winner()
        while winner != -1:
            move = self._select_move_in_simulation(copy)
            copy = copy.simulate_move(move)
            winner = copy.check_winner()

        if winner == starting_player:
            # if the child won, the parent must be penalized
            return -1
        else:
            # otherwise give him a big hug, parents deserve them
            return 1
    
    def _backpropagate(self, node: "MCTNode", score: Literal['-1', '1']) -> None:
        if score > 0:
            node.utility += score
        node.count += 1
        
        if node.parent:
            self._backpropagate(node.parent, -score)

    
if __name__ == "__main__":
    from helper import evaluate
    from main import RandomPlayer
    m = MCTSPlayer(500)
    evaluate(m, RandomPlayer(), 10, True)
