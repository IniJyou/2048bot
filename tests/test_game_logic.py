from copy import deepcopy

import numpy as np
import pytest

from game_logic import DOWN, LEFT, RIGHT, UP, Game2048, simulate_move


@pytest.mark.parametrize("row, expected, reward", [
    ([2, 2, 2, 2], [4, 4, 0, 0], 8),
    ([2, 2, 4, 0], [4, 4, 0, 0], 4),
    ([0, 2, 0, 2], [4, 0, 0, 0], 4),
    ([2, 2, 4, 4], [4, 8, 0, 0], 12),
    ([4, 4, 8, 8], [8, 16, 0, 0], 24),
    ([0, 0, 0, 0], [0, 0, 0, 0], 0),
    ([2, 4, 8, 16], [2, 4, 8, 16], 0),
])
def test_line_merges_once(row, expected, reward):
    assert Game2048()._move_left(row) == (expected, reward)


@pytest.mark.parametrize("action, target", [
    (UP, (0, 1)), (DOWN, (3, 1)), (LEFT, (1, 0)), (RIGHT, (1, 3)),
])
def test_directions(action, target):
    board = np.zeros((4, 4), dtype=int)
    board[1, 1] = 2
    original = board.copy()
    result = simulate_move(board, action)
    assert result.board[target] == 2
    assert result.board.sum() == 2
    assert result.reward == 0
    assert result.changed
    assert np.array_equal(board, original)
    assert not np.shares_memory(result.board, board)


def reference_move(board, action):
    """Independent compact/merge reference, without animation bookkeeping."""
    source = board if action in (LEFT, RIGHT) else board.T
    reverse = action in (RIGHT, DOWN)
    rows, total = [], 0
    for row in source:
        values = [int(value) for value in (row[::-1] if reverse else row) if value]
        output = []
        while values:
            value = values.pop(0)
            if values and values[0] == value:
                values.pop(0)
                value *= 2
                total += value
            output.append(value)
        output += [0] * (len(row) - len(output))
        rows.append(output[::-1] if reverse else output)
    result = np.array(rows)
    return (result if action in (LEFT, RIGHT) else result.T), total


def test_randomized_moves_against_reference():
    rng = np.random.default_rng(92)
    for _ in range(250):
        board = rng.choice([0, 2, 4, 8, 16, 32], size=(4, 4))
        for action in range(4):
            expected, reward = reference_move(board, action)
            result = simulate_move(board, action, track_motion=True)
            assert np.array_equal(result.board, expected)
            assert result.reward == reward
            assert result.board.sum() == board.sum()
            assert len(result.motions) == np.count_nonzero(board)
            destinations = {}
            for motion in result.motions:
                destinations[motion.destination] = destinations.get(motion.destination, 0) + motion.value
            for destination, value in destinations.items():
                assert result.board[destination] == value
            assert sum(int(result.board[cell]) for cell in result.merged_cells) == reward


def test_animation_tracks_both_merge_sources_and_afterstate():
    board = np.zeros((4, 4), dtype=int)
    board[0] = [0, 2, 0, 2]
    result = simulate_move(board, LEFT, track_motion=True)
    assert [(m.source, m.destination, m.value) for m in result.motions] == [
        ((0, 1), (0, 0), 2), ((0, 3), (0, 0), 2),
    ]
    assert result.merged_cells == ((0, 0),)
    assert result.board[0, 0] == 4
    assert np.count_nonzero(result.board) == 1


def test_legal_actions_preserve_board_identity_score_and_rng():
    game = Game2048(seed=4)
    game.board[:] = 0
    game.board[0, 0] = 2
    reference = game.board
    state = deepcopy(game)
    assert game.legal_actions() == [DOWN, RIGHT]
    assert game.board is reference
    assert np.array_equal(game.board, state.board)
    assert game.score == state.score
    assert game.rng.bit_generator.state == state.rng.bit_generator.state


def test_noop_does_not_spawn_or_consume_randomness():
    game = Game2048(seed=4)
    game.board[:] = 0
    game.board[0, 0] = 2
    state = deepcopy(game)
    board, reward, done = game.step(LEFT)
    assert np.array_equal(board, state.board)
    assert game.rng.bit_generator.state == state.rng.bit_generator.state
    assert reward == 0 and not done
    assert game.score == state.score


@pytest.mark.parametrize("action", [-1, 4, None, "left", 0.0, True])
def test_invalid_action_is_rejected_before_state_changes(action):
    game = Game2048(seed=8)
    state = deepcopy(game)
    with pytest.raises(ValueError):
        game.step(action)
    assert np.array_equal(game.board, state.board)
    assert game.rng.bit_generator.state == state.rng.bit_generator.state


def test_snapshots_restore_score_terminal_and_rng():
    game = Game2048(seed=15)
    game.board[0] = [2, 2, 0, 0]
    state = game.get_state()
    first, first_reward, first_done = game.step(LEFT)
    first_rng = deepcopy(game.rng.bit_generator.state)
    game.set_state(state)
    repeated, reward, done = game.step(LEFT)
    assert np.array_equal(first, repeated)
    assert (reward, done) == (first_reward, first_done)
    assert game.rng.bit_generator.state == first_rng
    state.board[:] = 0
    assert game.board.any()


def test_full_board_is_not_always_terminal():
    game = Game2048()
    game.board[:] = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]
    assert not game.has_legal_moves()
    game.step(LEFT)
    assert game.game_over and game.legal_actions() == []
    game.reset()
    game.board[:] = [[2, 2, 4, 8], [4, 8, 16, 32], [8, 16, 32, 64], [16, 32, 64, 128]]
    assert game.has_legal_moves()
    assert LEFT in game.legal_actions()


def test_random_games_preserve_score_and_spawn_invariants():
    rng = np.random.default_rng(4)
    for seed in range(10):
        game = Game2048(seed=seed)
        for _ in range(5000):
            if game.game_over:
                break
            before_sum, before_score = int(game.board.sum()), game.score
            board, reward, done = game.step(int(rng.choice(game.legal_actions())))
            assert int(board.sum()) - before_sum in (2, 4)
            assert game.score == before_score + reward
            assert done == (not game.has_legal_moves())
        else:
            pytest.fail("random game did not terminate")
