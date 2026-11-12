import numpy as np
import pygame
import pytest

from game_logic import LEFT, RIGHT
from gui import GameApp, Preferences


@pytest.fixture
def app():
    result = GameApp(storage_path=None, seed=20)
    result.game.board[:] = 0
    result.game.board[0] = [0, 2, 0, 2]
    return result


def click(app, name):
    app.handle_event(pygame.event.Event(
        pygame.MOUSEBUTTONDOWN, button=1, pos=app.buttons()[name].rect.center
    ))


def key(app, value, mod=0):
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=value, mod=mod))


def test_undo_restores_randomness_and_is_single_use(app):
    original = app.game.get_state()
    app.play_human(LEFT)
    board, score = app.game.board.copy(), app.game.score
    assert app.animation is not None
    assert app.undo()
    assert app.animation is None and app.pending_action is None
    assert np.array_equal(app.game.board, original.board)
    assert app.game.score == original.score
    assert app.game.rng.bit_generator.state == original.rng_state
    assert not app.undo()
    app.play_human(LEFT)
    assert np.array_equal(app.game.board, board)
    assert app.game.score == score
    assert app.preferences.best_score == score


def test_noop_preserves_undo(app):
    app.set_speed(2)
    app.play_human(LEFT)
    previous = app.undo_state
    # Force a known no-op position without altering the undo slot.
    app.game.board[:] = 0
    app.game.board[0, 0] = 2
    assert not app.play_human(LEFT)
    assert app.undo_state is previous


def test_highest_speed_has_no_animation_or_queue(app):
    app.set_speed(2)
    assert app.play_human(LEFT)
    assert app.animation is None
    assert app.play_human(RIGHT)
    assert app.animation is None and app.pending_action is None


def test_animation_keeps_one_queued_action_and_menu_cancels_it(app):
    app.play_human(LEFT)
    first_score = app.game.score
    app.play_human(RIGHT)
    assert app.pending_action == RIGHT and app.game.score == first_score
    key(app, pygame.K_ESCAPE)
    assert app.view == "menu"
    assert app.animation is None and app.pending_action is None
    board = app.game.board.copy()
    app.update(10)
    key(app, pygame.K_LEFT)
    assert np.array_equal(app.game.board, board)


def test_queued_action_executes_once_after_animation(app):
    app.play_human(LEFT)
    app.play_human(RIGHT)
    app.update(1)
    assert app.animation is not None
    assert app.pending_action is None
    app.update(1)
    assert app.animation is None
    assert app.pending_action is None


def test_menu_buttons_speed_about_reset_and_exit(app):
    click(app, "menu")
    assert app.view == "menu"
    assert not app.buttons()["undo"].enabled
    assert set(app.buttons()) == {
        "undo", "menu", "speed_0", "speed_1", "speed_2", "about", "reset", "back", "exit"
    }
    click(app, "speed_2")
    assert app.preferences.speed_index == 2
    click(app, "about")
    assert app.view == "about"
    key(app, pygame.K_ESCAPE)
    assert app.view == "menu"
    click(app, "back")
    assert app.view == "game"
    app.play_human(LEFT)
    assert app.undo_state is not None
    click(app, "menu")
    click(app, "reset")
    assert app.view == "game" and app.undo_state is None
    assert app.game.score == 0 and np.count_nonzero(app.game.board) == 2
    click(app, "menu")
    click(app, "exit")
    assert not app.running


def test_keyboard_undo_and_reset(app):
    key(app, pygame.K_LEFT)
    key(app, pygame.K_z, pygame.KMOD_CTRL)
    assert app.game.score == 0 and app.undo_state is None
    key(app, pygame.K_LEFT)
    key(app, pygame.K_r)
    assert app.game.score == 0 and app.animation is None and app.undo_state is None


def test_game_over_can_be_undone(app):
    app.play_human(LEFT)
    app.finish_animation()
    app.game.game_over = True
    app.render()
    click(app, "over_undo")
    assert not app.game.game_over and app.undo_state is None


def test_preferences_survive_restart_and_corrupt_files(tmp_path):
    path = tmp_path / "preferences.json"
    preferences = Preferences(best_score=12345, speed_index=2)
    preferences.save(path)
    assert Preferences.load(path) == preferences
    path.write_text('{"best_score": -4, "speed_index": 9}', encoding="utf-8")
    assert Preferences.load(path) == Preferences()
    for value in ["broken json", "[]", "null", '{"speed_index": true}']:
        path.write_text(value, encoding="utf-8")
        assert Preferences.load(path) == Preferences()


def test_render_each_view_and_mid_animation(app):
    app.play_human(LEFT)
    app.update(0.04)
    moving = pygame.image.tobytes(app.render(), "RGB")
    app.update(0.13)
    merging = pygame.image.tobytes(app.render(), "RGB")
    assert moving != merging
    app.update(1)
    for view in ("game", "menu", "about"):
        app.view = view
        assert app.render().get_size() == (560, 820)
    app.view = "game"
    app.game.score = 99999999
    app.game.board[0] = [2048, 4096, 16384, 131072]
    assert app.render().get_size() == (560, 820)


def test_letterboxed_mouse_hit_testing(app):
    app.screen = pygame.Surface((900, 700))
    button = app.buttons()["menu"].rect
    scale = min(900 / 560, 700 / 820)
    position = ((900 - 560 * scale) / 2 + button.centerx * scale,
                (700 - 820 * scale) / 2 + button.centery * scale)
    app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=position))
    assert app.view == "menu"
