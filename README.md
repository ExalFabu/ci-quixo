# Computational Intelligence - Quixo
# Authors:
- Alexandro Buffa - S316999


## Description
The game Quixo is a Tic-Tac-Toe variant, played on a five-by-five board of cubes with two players or teams. On a player's turn, they select a blank cube or a cube with their symbol on it that is at the edge of the board. If a blank cube was selected, the cube is turned to be the player's symbol (either an X or O). The game ends when one player gets five in a row.

## What I've Done
- Minmax
  - Alpha-Beta Pruning
  - Hash-Tables
  - Symmetries
- Montecarlo Tree Search
  - Random
  - w/ heuristic

## Navigate through the code

- `custom_game.py`: Wrapper around Game class, with some utility methods and symmetry (canonical representation) handling
- `minmax.py` and `mcts.py`: Player's files
- `__main__.py`: containis the code to perform the evaluation

## Possible Improvements

- [ ] Minmax w/ RankCut
- [ ] Minmax w/ Singular Moves (should be easy and fast-enough using Hash Tables)
- [ ] Minmax w/ Parallelization
