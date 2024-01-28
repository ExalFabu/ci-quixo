from typing import TYPE_CHECKING, Literal, Union
try:
    from game import Player, Game
    from custom_game import CustomGame, POSSIBLE_MOVES
    if TYPE_CHECKING:
        from custom_game import CompleteMove
except: 
    from .game import Player, Game
    from .custom_game import CustomGame, POSSIBLE_MOVES
    if TYPE_CHECKING:
        from .custom_game import CompleteMove

import numpy as np
from collections import defaultdict
import random
import time



class MinMaxPlayer(Player):
    """ Minimax Player with alpha-beta pruning (togglable) and a hash-table to store previously evaluated states. 

        There are 4 possible pruning 'levels' (explained in detail below), i believe the best tradeoff between pruning and speed is level 1,
        going at a deeper level is just too much time wasted due to the time required to process the (ineffiently implemented) symmetries. 
        To have an understanding of the difference of time there is a bencharmking function that shows it (see `custom_game.test_benchmark_symmetries`), spoiler: +2400%
    """

    def __init__(
        self,
        max_depth: int = 2,
        *,
        alpha_beta: bool = True,
        pruning: Literal["0", "1", "2", "3"] = 1,
        htable: bool = True,

    ) -> None:
        """Init

        Args:
            max_depth (int, optional): Tree depth. Defaults to 2.
            alpha_beta (bool, optional): Whether to use the Alpha-Beta pruining. Defaults to True.
            pruning (Literal['0', '1', '2', '3'] , optional): Pruning level. Defaults to 1.
                This pruning level determines the amount of pre-filtering done to the MinMax tree (i.e. how many children a node has)
                0: Consider only valid moves
                1: Consider only valid moves that land on distinct boards (purge moves that would land on a board that is already covered by another move)
                2: Consider only valid moves that land on distinct *canonical* boards (purge moves that would land on the same equivalence class of already covered boards)
                3: Same as 2, plus we filter the boards that we have already covered on a lower depth (where the lowest is the root)
                    This is done because it is possible, with a sufficiently high `max_depth`, to loop into an already covered board, 
                    and if I have encountered it at a lower depth, it means that that evaluation has more information than I can ever hope to achieve, meaning it's useless 
                    to expand this subtree
            htable (bool, optional): Whether to use an hash-table to save and use already evaluated states. Defaults to True.
        """
        super().__init__()
        
        self.max_depth = 2 if max_depth is None else max_depth
        self.use_alpha_beta_pruning = alpha_beta
        self.pruning_level = pruning
        self.use_htable = htable
        
        self.history: dict[str, "CompleteMove"] = dict()
        """Hash-Table but only for complete moves, always enabled"""
        self.htable: dict[
            str, dict[tuple[Literal["l", "h"], int], float]
        ] = defaultdict(lambda: defaultdict(float))
        """Hash Table for intermediate states, enabled with `htable` flag"""

        self._stats = defaultdict(int)
        """Used to gather some basic stats and counters"""

    @property
    def short_name(self) -> str:
        '''Short Name used in pictures'''
        return f"MinMax({'AB, ' if self.use_alpha_beta_pruning else ''}D{self.max_depth}, P{self.pruning_level}{', H' if self.use_htable else ''})"

    @property
    def name(self) -> str:
        '''Full Name'''
        return f"MinMax(depth={self.max_depth}, alpha_beta={self.use_alpha_beta_pruning}, pruning={self.pruning_level}, use_htable={self.use_htable})"

    def make_move(self, game: Game) -> "CompleteMove":
        start = time.time()
        cg = CustomGame.from_game(game)
        best_move = self._minmax(cg)
        if best_move is None or not cg.is_valid(best_move):
            self._stats["EVAL-invalidmove"] += 1
            best_move = random.choice(cg.valid_moves())
        else:
            self._stats['evals'] += 1
            self._stats['evals-ms'] += (time.time() - start)
            
        
        return best_move

    def search_in_htable(
        self, game: "CustomGame", curr_depth: int, curr_side: Literal["l", "h"]
    ) -> Union[float, None]:
        """Searches the move in the hash-table
        Look for states explored previously prioritizing the ones that have been visited with a lower depth 
        (meaning it explored more states, and thus has more insights), exploiting also the states that have been visited by the other players (*-1)


        Args:
            game (CustomGame): game we are looking for
            curr_depth (int): current depth
            curr_side (Literal[l,h]): Side at which we are on (l = low = min | h = high = max)

        Returns:
            Union[float, None]: The value stored in the hash table if found, None otherwise (or if htable disabled)
        """
        if not self.use_htable or str(game) not in self.htable:
            self._stats["HTABLE-MISS"] += 1
            return None

        visited = self.htable[str(game)]
        samesies = defaultdict(float)
        '''stored here are the visited states played on the same side'''
        contries = defaultdict(float)
        '''stored here are the visited states played on the opponent side'''

        for key, value in visited.items():
            side, depth = key
            if side == curr_side and depth <= curr_depth:
                samesies[depth] = value
            elif side != curr_side and depth <= curr_depth:
                # note here that if we are at an opponent side, we flip the value (leveraging the fact that our heuristic can do so)
                contries[depth] = -value

        # If there are some match save the (depth, value) of it, 
        if len(samesies) != 0: 
            sms_dv = min(samesies.keys())
            sms_dv = (sms_dv, samesies[sms_dv])
        else:
            #otherwise continue with an impossible value that we will filter out later
            sms_dv = (self.max_depth +10, None)
        
        if len(contries) != 0:
            cnt_dv = min(contries.keys())
            cnt_dv = (cnt_dv, contries[cnt_dv])
        else:
            cnt_dv = (self.max_depth +10, None)

        # Get the (depth, value) that has the lowest depth (meaning more insight)
        dv = sms_dv if sms_dv[0] < cnt_dv[0] else cnt_dv

        # If a match is really found (and it is not an impossible thing that we previously set), return the value of that move
        if dv[0] <= self.max_depth:
            self._stats["HTABLE-HIT"] += 1
            self._stats[f"HTABLE-HIT-{dv[0]}/{curr_depth}"] += 1
            return dv[1]
        
        self._stats["HTABLE-MISS"] += 1
        return None

    def put_in_htable(
        self,
        game: "CustomGame",
        curr_depth: int,
        curr_side: Literal["l", "h"],
        value: float,
    ) -> None:
        """Save move in htable if enabled

        Args:
            game (CustomGame): Game
            curr_depth (int): current depth
            curr_side (Literal[l, h]): Side we are currently on
            value (float): value of that game
        """

        if self.use_htable:
            self.htable[str(game)][(curr_side, curr_depth)] = value

    def _minmax(self, game: "CustomGame") -> "CompleteMove":
        visited_list: list[set[str]] = [set() for _ in range(self.max_depth)]
        """Store here the states that i have visited while making a move (used only with pruning_level == 4)"""
        
        # Store the player that is to move, used in various assertions when I was trying to figure out what was wrong,
        # keeping it because it makes sense 
        whoami = game.get_current_player()

        def moves_getter(game: "CustomGame", depth: int) -> list[tuple["CompleteMove", "CustomGame"]]:
            """Utility function used to get the moves when expanding a node, used both in max and min side"""

            self._stats["MOVES-THEORETICAL"] += 44  # length of POSSIBLE_MOVES

            if self.pruning_level == 0:
                # Only valid moves
                moves = game.valid_moves(None, False, False)
            elif self.pruning_level == 1:
                # filter the moves that land on a board already covered
                moves = game.valid_moves(None, True, False)
            else: # both 2 and 3
                # filter the moves that land on a board already covered (using symmetries)
                moves = game.valid_moves(None, True, True)

            games = [game.simulate_move(move) for move in moves]
            move_n_games = list(zip(moves, games))
            
            if self.pruning_level == 3:
                # Filter also the boards that we already covered at a lower depth
                visited_list[depth].union(set([str(it) for it in games]))
                already_visited = set([game for d in range(0, depth) for game in visited_list[d]])
                """Already visited games at a lower depth"""
                # For stats purposes
                _pre = len(move_n_games)
                move_n_games = [it for it in move_n_games if str(it[1]) not in already_visited]
                # For stats purposes
                _post = len(move_n_games)
                self._stats["PRUNING3-DIFF"] += _pre-_post # this is always 0.. Is depth 2 enough for a loop? Maybe 3 might
            
            self._stats["MOVES-ACTUAL"] += len(move_n_games)
            return move_n_games
            
        def min_side(
            self: "MinMaxPlayer", game: "CustomGame", alpha: int, beta: int, depth: int
        ) -> int:
            assert game.current_player_idx == 1-whoami, "Something went awfully wrong"

            htable_value = self.search_in_htable(game, depth, "l")
            if htable_value:
                return htable_value

            winner = game.check_winner()
            if (self.max_depth is not None and depth >= self.max_depth) or winner != -1:
                score = -1 * game.score # We want the score as if I'm the other player (thus *-1)
                self.put_in_htable(game, depth, "l", score)
                return score

            min_found = np.infty

            for _, copy in moves_getter(game, depth):
                min_found = min(min_found, max_side(self, copy, alpha, beta, depth + 1))
                if alpha >= min_found and self.use_alpha_beta_pruning:
                    break
                beta = min(beta, min_found)

            self.put_in_htable(game, depth, "l", min_found)
            return min_found

        def max_side(
            self: "MinMaxPlayer", game: "CustomGame", alpha: int, beta: int, depth: int
        ) -> int:
            assert game.current_player_idx == whoami, "Something went awfully wrong"

            htable_value = self.search_in_htable(game, depth, "h")
            if htable_value:
                return htable_value

            winner = game.check_winner()
            if (self.max_depth is not None and depth >= self.max_depth) or winner != -1:
                score = game.score
                self.put_in_htable(game, depth, "h", score)
                return score

            max_found = -np.infty

            for _, copy in moves_getter(game, depth):
                max_found = max(max_found, min_side(self, copy, alpha, beta, depth + 1))
                if max_found >= beta and self.use_alpha_beta_pruning:
                    break
                alpha = max(alpha, max_found)

            self.put_in_htable(game, depth, "h", max_found)
            return max_found

        # Start MinMax 

        best_move = None
        alpha, beta = -np.inf, np.inf

        if str(game) in self.history:
            self._stats["cache-hit"] += 1
            return self.history[str(game)]

        for move, copy in moves_getter(game, 0):
            min_score = min_side(self, copy, alpha, beta, 1)
            if min_score > alpha:
                alpha = min_score
                best_move = move
        self._stats["EVALS"] += 1
        self.history[str(game)] = best_move
        self.put_in_htable(game, 0, "h", alpha)
        return best_move


    @property 
    def _avg_time(self):
        if self._stats['evals'] == 0:
            return 0
        return self._stats['evals-ms'] / self._stats['evals']
    

    @property
    def stats(self) -> dict[str, str]:
        """Pretty Print relevant stats

        Returns:
            dict[str, str]: pretty printed stats
        """
        # actual moves performed and total number of moves that would've been performed without any pruning at all (not even lvl 0)
        am, thm = self._stats["MOVES-ACTUAL"], self._stats["MOVES-THEORETICAL"]

        pp = {
            "Average time per move": f"{self._avg_time:.2f}s",  
            f"Pruning lvl. {self.pruning_level} discount": f"{(1-(am/thm)):.2%}",
            "Total Moves performed": self._stats["evals"]
        }
        if self._stats["EVAL-invalidmove"] != 0:
            # This should never happen, but as we say where I come from "pi na mano..." 
            # (which very roughly translates to: "better safe than sorry")
            pp['Invalid Moves performed'] = self._stats["EVAL-invalidmove"]
        if self.use_htable:
            hitratio = self._stats["HTABLE-HIT"] / (self._stats['HTABLE-MISS'] + self._stats['HTABLE-HIT'])
            # Ratio of games found over all games explored
            pp["HashTable HitRatio"] = f"{hitratio:.3%}"
        return pp
    
if __name__ == "__main__":
    try:
        from helper import evaluate
    except:
        from .helper import evaluate

    from pprint import pprint

    mf = MinMaxPlayer(2, pruning=0, htable=False)
    evaluate(mf, None, 50, True)
    pprint(mf.stats, sort_dicts=False)
