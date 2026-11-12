"""2048 rules, independent of the GUI and suitable for search/training."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np


UP, DOWN, LEFT, RIGHT = range(4)


@dataclass(frozen=True)
class TileMotion:
    value: int
    source: tuple[int, int]
    destination: tuple[int, int]


@dataclass(frozen=True)
class SpawnedTile:
    position: tuple[int, int]
    value: int


@dataclass(frozen=True)
class MoveResult:
    board: np.ndarray
    reward: int
    changed: bool
    motions: tuple[TileMotion, ...] = ()
    merged_cells: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class GameState:
    board: np.ndarray
    score: int
    game_over: bool
    rng_state: dict


def _validate_action(action):
    if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
        raise ValueError("action must be an integer: 0=up, 1=down, 2=left, 3=right")
    if action not in (UP, DOWN, LEFT, RIGHT):
        raise ValueError("action must be one of 0, 1, 2, 3")


def simulate_move(board, action, *, track_motion=False):
    """Return an independent afterstate, without spawning tiles or using RNG.

    ``track_motion`` is only needed by the human-facing GUI. Search and AI
    can use the default to avoid constructing animation records.
    """
    _validate_action(action)
    board = np.asarray(board)
    if board.ndim != 2 or board.shape[0] != board.shape[1]:
        raise ValueError("board must be a square two-dimensional array")
    size = board.shape[0]
    result = np.zeros_like(board)
    reward = 0
    motions, merged_cells = [], []

    for lane in range(size):
        if action == UP:
            coordinates = [(row, lane) for row in range(size)]
        elif action == DOWN:
            coordinates = [(row, lane) for row in range(size - 1, -1, -1)]
        elif action == LEFT:
            coordinates = [(lane, col) for col in range(size)]
        else:
            coordinates = [(lane, col) for col in range(size - 1, -1, -1)]

        occupied = [position for position in coordinates if board[position] != 0]
        source_index, target_index = 0, 0
        while source_index < len(occupied):
            source = occupied[source_index]
            value = int(board[source])
            destination = coordinates[target_index]
            merging = (
                source_index + 1 < len(occupied)
                and board[occupied[source_index + 1]] == value
            )
            result[destination] = value * 2 if merging else value
            if track_motion:
                motions.append(TileMotion(value, source, destination))
            if merging:
                reward += value * 2
                if track_motion:
                    motions.append(TileMotion(value, occupied[source_index + 1], destination))
                    merged_cells.append(destination)
                source_index += 2
            else:
                source_index += 1
            target_index += 1

    return MoveResult(
        result, reward, not np.array_equal(result, board),
        tuple(motions), tuple(merged_cells),
    )


class Game2048:
    def __init__(self, size=4, seed=None):
        if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 2:
            raise ValueError("size must be an integer of at least 2")
        self.size = int(size)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.board = np.zeros((self.size, self.size), dtype=np.int64)
        self.score = 0
        self.game_over = False
        self.last_move = None
        self.last_spawn = None
        self._add_random_tile()
        self._add_random_tile()
        return self.board.copy()

    def _add_random_tile(self):
        empty_cells = np.argwhere(self.board == 0)
        if not len(empty_cells):
            return None
        row, col = empty_cells[self.rng.integers(len(empty_cells))]
        position = (int(row), int(col))
        value = 2 if self.rng.random() < 0.9 else 4
        self.board[position] = value
        return SpawnedTile(position, value)

    def _move_left(self, row):
        """Keep the original single-row helper for exercises and callers."""
        board = np.zeros((self.size, self.size), dtype=np.int64)
        board[0] = row
        result = simulate_move(board, LEFT)
        return result.board[0].tolist(), result.reward

    def step(self, action, *, capture_animation=False):
        """0=up, 1=down, 2=left, 3=right; return (board, reward, done).

        No undo history is kept here. The GUI owns its single human undo slot.
        """
        _validate_action(action)
        self.last_move = None
        self.last_spawn = None
        if self.game_over:
            return self.board.copy(), 0, True

        result = simulate_move(self.board, action, track_motion=capture_animation)
        if result.changed:
            self.board[...] = result.board
            self.score += result.reward
            self.last_spawn = self._add_random_tile()
            if capture_animation:
                self.last_move = result
        self.game_over = not self.has_legal_moves()
        return self.board.copy(), result.reward, self.game_over

    def has_legal_moves(self):
        return bool(
            np.any(self.board == 0)
            or np.any(self.board[:, :-1] == self.board[:, 1:])
            or np.any(self.board[:-1, :] == self.board[1:, :])
        )

    def legal_actions(self):
        if self.game_over:
            return []
        return [action for action in range(4) if simulate_move(self.board, action).changed]

    def get_state(self):
        """Capture board, score and randomness for a reproducible human undo."""
        return GameState(
            self.board.copy(), self.score, self.game_over,
            deepcopy(self.rng.bit_generator.state),
        )

    def set_state(self, state):
        if state.board.shape != (self.size, self.size):
            raise ValueError("snapshot board size does not match the game")
        self.board[...] = state.board
        self.score = int(state.score)
        self.game_over = bool(state.game_over)
        self.rng.bit_generator.state = deepcopy(state.rng_state)
        self.last_move = None
        self.last_spawn = None
