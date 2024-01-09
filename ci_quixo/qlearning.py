from custom_game import CustomGame, POSSIBLE_MOVES, CompleteMove
from game import Game, Move
from copy import deepcopy
from collections import defaultdict
from main import Player
from typing import DefaultDict
from dataclasses import dataclass, field
from tqdm.auto import trange, tqdm
import random, dill
import numpy as np
from main import RandomPlayer

# QTable Structure
# [key: board hash (str)]: {
#   [complete_move]: float
# }

QTable = DefaultDict[str, DefaultDict["CompleteMove", float]]
"""QTable structure
{
    [key: board hash (str)]: {
        [complete_move]: float
        ...
        [complete_move]: float
    },
}
"""


def clamp(value, min_, max_):
    """Clamp value between min_ and max_"""
    return min(max(value, min_), max_)


@dataclass
class QLearning(Player):
    
    qtable: QTable = field(
        default_factory=lambda: defaultdict(QLearning.__inner_defdict_builder)
    )
    learning_rate: float = field(default=0.1)
    discount_rate: float = field(default=0.99)
    exploration_rate: float = field(default=1)
    min_exploration_rate: float = field(default=0.01)
    exploration_decay_rate: float = field(default=2e-4)
    num_of_episodes: int = field(default=1_000)
    stats: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @staticmethod
    def DUMP_FILENAME():
        return "qlearning.npy"


    def reward(self, board: "CustomGame", move: "CompleteMove", plind: int) -> float:
        target = board.simulate_move(move)
        winner = target.check_winner()
        if winner != -1:
            return target.score * (1 if target.current_player_idx == plind else -1)
        else:
            return target.score

    def training_move_chooser(self, game: "CustomGame") -> "CompleteMove":
        if random.uniform(0, 1) > self.exploration_rate:
            # exploit
            self.stats["exploit"] += 1
            if str(game) in self.qtable:
                moves_dict = self.qtable[str(game)]
                if not len(moves_dict.items()) == 0:
                    self.stats["exploit_ok"] += 1
                    return max(moves_dict.items(), key=lambda it: it[1])[0]
        # explore or nothing to exploit
        self.stats["explore"] += 1
        return random.choice(game.valid_moves())

    def train(self: "QLearning", opponent: "Player" = None, verbose: bool = False):
        if not verbose:
            vprint = lambda x: None
        else:
            vprint = print
        if opponent is None:
            opponent = RandomPlayer()
        pbar = trange(
            self.num_of_episodes,
            unit="episode",
            desc=f"Training QLearning",
        )
        for episode in pbar:
            game = CustomGame()
            whoami = 0 if episode % 2 == 0 else 1
            move: "CompleteMove"
            while game.check_winner() == -1:
                if whoami == game.next_move_for:
                    # QLeraning Turn
                    # Step 1: Go to canonical state
                    original_board = deepcopy(game)
                    canon_board, canon_idx = game.to_canon()
                    # Step 2: Select the move
                    move = self.training_move_chooser(canon_board)
                    reward = self.reward(canon_board, move, whoami)
                    after_move_board = canon_board.simulate_move(move)
                    # Step 3: Update qtable
                    self.qtable[str(canon_board)][move] *= 1 - self.learning_rate
                    self.qtable[str(canon_board)][move] += self.learning_rate * (
                        reward
                        + self.discount_rate
                        * (-np.max([0, *list(self.qtable[str(after_move_board)].values())]))
                    )
                    # Step 4: Return to normal game in order not to affect opponent's behaviour
                    move = CustomGame.convert_canon_move(
                        canon_board, move, original_board
                    )
                    game = original_board.simulate_move(move)
                else:
                    # Opponent Turn
                    opponent_move = None
                    while (
                        opponent_move is None
                        or opponent_move not in game.valid_moves(None, False)
                    ):
                        opponent_move = opponent.make_move(game)
                    game = game.simulate_move(opponent_move)

            reward = (5**5) * (1 if game.check_winner() == whoami else -1)
            self.qtable[str(canon_board)][move] *= 1 - self.learning_rate
            self.qtable[str(canon_board)][move] += self.learning_rate * (
                reward
                + self.discount_rate * (-np.max([0, *list(self.qtable[str(after_move_board)].values())]))
            )

            self.exploration_rate = clamp(
                np.exp(-self.exploration_decay_rate * episode),
                self.min_exploration_rate,
                1,
            )
            if (
                episode
                % clamp(int(self.num_of_episodes / 100), 1, self.num_of_episodes)
                == 0
            ):
                pbar.set_postfix({"Explored": len(self.qtable.keys())})

    def make_move(self, game: Game) -> "CompleteMove":
        self.stats["EVAL-moves_requested"] += 1
        custom = CustomGame.from_game(game)
        canon, _ = custom.to_canon()
        best_move = None
        if str(canon) in self.qtable:
            moves_dict = self.qtable[str(canon)]
            if not len(moves_dict.items()) == 0:
                move = max(moves_dict.items(), key=lambda it: it[1])[0]
                if move in canon.valid_moves(None, False):
                    self.stats["EVAL-found"] += 1
                    best_move = move
        if best_move is None:
            return random.choice(custom.valid_moves())
        move = CustomGame.convert_canon_move(canon, best_move, custom)
        if move in custom.valid_moves(None, False):
            return move
        else:
            self.stats["EVAL-somehow went here? investigate"]
            return random.choice(custom.valid_moves())

    @staticmethod
    def __inner_defdict_builder() -> DefaultDict["CompleteMove", float]:
        return defaultdict(float)
    
    def dump(self: "QLearning"):
        with open(QLearning.DUMP_FILENAME(), "wb") as f:
            dill.dump(self.qtable, f)

    def load(self: "QLearning") -> bool:
        try:
            with open(QLearning.DUMP_FILENAME(), "rb") as f:
                self.qtable = dill.load(f)
            return True
        except:
            return False

if __name__ == "__main__":
    from helper import evaluate
    episodes = 10_000
    q = QLearning(num_of_episodes=episodes)
    q.load()
    action: str = None
    while action is None or not action.isdigit():
        action = input("1. Train All Over Again\n2. Train on Top of backup\n3. Use Backup only\n> ").strip()
    action = int(action)
    if action != 3:
        if action == 1:
            q = QLearning(num_of_episodes=episodes)
        q.train(None, True)
        q.dump()
    evaluate(q, None, 100, True)