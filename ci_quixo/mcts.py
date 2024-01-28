from typing import TYPE_CHECKING, Literal
try:
    from game import Game, Move, Player
    from custom_game import CustomGame
    if TYPE_CHECKING:
        from custom_game import CompleteMove
except:
    from .game import Game, Move, Player
    from .custom_game import CustomGame
    if TYPE_CHECKING:
        from .custom_game import CompleteMove

import numpy as np, random
from dataclasses import dataclass, field
from collections import defaultdict
from copy import deepcopy
import time
from tqdm.auto import trange, tqdm

# implementation inspired from https://github.com/aimacode/aima-python/blob/61d695b37c6895902081da1f37baf645b0d2658a/games4e.py#L178

@dataclass
class MCTNode:
    """Monte Carlo Tree Node

    Wrapper for a node of the MCTS that contains the utility and count values, parent and children references
    """
    
    state: "CustomGame" = field()
    parent: "MCTNode" = field()
    constant_factor: float = field(default=1.4)
    utility: int = field(default=0, init=False)
    count: int = field(default=0, init=False)
    children: dict["CompleteMove", "MCTNode"] = field(default_factory=lambda: dict(), init=False)

    def ucb(self, constant_factor = None):
        """Upper Confidence Bound 1 applied to trees

        Args:
            constant_factor (float, optional): exploration parameter. Defaults to `sqrt(2)`.

        Returns:
            float: `self.utility/self.count + constant_factor * sqrt(log(parent.count)/(self.count))`. If it has never been visited, returns `+inf`
        """
        if constant_factor is None:
            constant_factor = self.constant_factor
        
        if self.count == 0:
            return float("inf")
        return self.utility / self.count + constant_factor * (np.sqrt(np.log(self.parent.count) / self.count))

@dataclass
class MCTSPlayer(Player):
    """Monte Carlo Tree Search Player

    Disclaimer:
        Implementation took insipiration from looking at different sources, such as
         - [Artificial Intelligence: a Modern Approach](https://aima.cs.berkeley.edu/) and it's code [here](https://github.com/aimacode/aima-python/blob/61d695b37c6895902081da1f37baf645b0d2658a/games4e.py#L178)
         - [Monte Carlo Tree Search - Wikipedia](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)
    """
    
    games: int = field(default=500)
    """Number of games to play for each move"""
    sim_heuristic: bool = field(default=False)
    """Whether to use an heuristic when simulating a node. 

    If disabled, the simulation is played random
    If enabled, it uses the same scoring function used for minmax to determine the best next move
    """

    progress: bool = field(default=False)
    """Show progress bar while playing.. used this when I discovered that it could loop while playing using heuristic (see stats.loop and stats.deeploop :'))"""
    
    _stats: dict[str, int] = field(default_factory=lambda: defaultdict(int), init=False)
    """Simple dict used to keep track of basic statistics, see property stats for a prettified version"""

    @property
    def short_name(self) -> str:
        """Used in graphs pictures"""
        return f"MCTS({'H' if self.sim_heuristic else 'R'}, {self.games})"
    
    @property
    def name(self) -> str:
        return f"MCTS(games={self.games}, use_heuristic_in_simulation={self.sim_heuristic})"
    
    def make_move(self, game: Game) -> tuple[tuple[int, int], Move]:
        start = time.time()
        root_cg = CustomGame.from_game(game)

        root = MCTNode(root_cg, None)
        if self.progress:
            range_games = trange(self.games, unit="games", leave=False)
        else:
            range_games = range(self.games)
        for _ in range_games:
            self.progress and range_games.set_postfix({"phase": "select"})
            leaf = self._select(root)
            
            self.progress and range_games.set_postfix({"phase": "expand"})
            child = self._expand(leaf)
            
            self.progress and range_games.set_postfix({"phase": "simulate"})
            score = self._simulate(child)
            
            self.progress and range_games.set_postfix({"phase": "backprop"})
            self._backpropagate(child, score)
        
        # The Best Move is the child of the root that has been visited the most
        best_move = max(root.children.items(), key=lambda it: it[1].count)[0]
        self._stats['evals'] += 1

        if best_move not in root_cg.valid_moves(None, False, False):
            self._stats['eval-invalid'] += 1
            best_move = random.choice(root_cg.valid_moves(None, False, False))
        else:
            self._stats['evals-ms'] += time.time()-start
        return best_move

    def _select(self, node: "MCTNode") -> "MCTNode":
        """Select Phase - Choose the leaf using UCB function"""
        if node.children:
            return self._select(max(node.children.values(), key=MCTNode.ucb))
        else:
            return node
        
    def _expand(self, node: "MCTNode") -> "MCTNode":
        if not node.children or node.state.check_winner() == -1:
            # If the node has no children and is not a terminal state, expand all the children
            node.children = {
                move: MCTNode(node.state.simulate_move(move), node)
                for move in node.state.valid_moves(None, False, False)
            }
        
        return self._select(node)
    
    def _select_move_in_simulation(self, game: "CustomGame", i: int = 0) -> tuple["CompleteMove", "CustomGame"]:
        """Move selector in simulation phase - What moves are going to be played?

        Args:
            game (CustomGame): Game board
            i (int): In case we are in a loop, start getting sub-optimal moves to escape
        
        Returns:
            tuple[CompleteMove, CustomGame]: Move and Game
        """

        if self.sim_heuristic:
            # If we are using an heuristic, sort them accordingly to the score of the landing state
            moves =  game.valid_moves(None, True, True)
            games = [game.simulate_move(move) for move in moves]
            
            mg = zip(moves, games)
            score_sorted_move_games = sorted(mg, key=lambda it: it[1].score)
            # Start escaping the loop
            return score_sorted_move_games[i % len(score_sorted_move_games)]
        else:
            # Play random
            move = random.choice(game.valid_moves(None, False, False))
            return move, game.simulate_move(move)

    def _simulate(self, node: "MCTNode") -> int:
        """Simulate Phase - Plays one single game"""

        starting_player = node.state.get_current_player()

        copy = deepcopy(node.state)
        winner = copy.check_winner()

        if self.progress:
            pbar = tqdm(None, desc="move", leave=False)

        # Used to detect "simple loops" (A and B play always the same move)
        last_moves = [None, None]
        dup_counter = 0
        
        # Used to detect "deep loops" (A and B land on a state that has been visited more than 50 times)
        visited: dict[str, int] = defaultdict(int)
        
        while winner != -1:
            curr_player = copy.get_current_player()

            if dup_counter > 40:
                # If we are in a simple loop, start playing other moves
                move, copy = self._select_move_in_simulation(copy, dup_counter-20)
                self._stats["loop-dodged"] += 1
            else:
                move, copy = self._select_move_in_simulation(copy)
            
            if last_moves[curr_player] == move:
                dup_counter += 1
            else:
                dup_counter = 0


            visited[str(copy)] += 1

            if visited[str(copy)] > 50:
                # Deep loop 
                self._stats["deeploop-dodged"] += 1
                move, copy = self._select_move_in_simulation(copy, visited[str(copy)]-50)

            last_moves[curr_player] = move

            self.progress and pbar.update(1)
            self.progress and pbar.set_postfix({"board": str(copy), "move": move})
            
            winner = copy.check_winner()

        if winner == starting_player:
            # if the child won, the parent must be penalized
            return -1
        else:
            # otherwise give him a big hug, parents deserve them
            return 1
    
    def _backpropagate(self, node: "MCTNode", score: Literal['-1', '1']) -> None:
        """Backpropagate till the root"""

        if score > 0:
            node.utility += score
        node.count += 1
        
        if node.parent:
            self._backpropagate(node.parent, -score)


    @property 
    def _avg_time(self):
        if self._stats['evals'] == 0:
            return 0
        return self._stats['evals-ms'] / self._stats['evals']
    
    @property
    def stats(self):
        """Pretty Printed stats"""
        return {
            "Average time per move": f"{self._avg_time:.2f}s",
            "Total Moves performed": self._stats['evals'],
            "Loops Dodged": self._stats['loop-dodged'],
            "Deep-Loop Dodged": self._stats['deeploop-dodged']
        }
    
if __name__ == "__main__":
    from helper import evaluate
    from main import RandomPlayer
    from pprint import pprint
    games_for_evaluation = 10
    mcts_depth = 500
    show_progress = False
    ###
    mr = MCTSPlayer(mcts_depth, False, show_progress)
    print("---\t---")
    print(f"MCTS({mcts_depth}) Simulating with random moves")
    evaluate(mr, RandomPlayer(), games_for_evaluation, True)
    pprint(mr.stats, sort_dicts=False)
    mh = MCTSPlayer(mcts_depth, True, show_progress)
    print("---\t---")
    print(f"MCTS({mcts_depth}) Simulating with heuristic")
    evaluate(mh, RandomPlayer(), games_for_evaluation, True)
    pprint(mh.stats, sort_dicts=False)
