"""A small Pygame front end. Run with: python gui.py"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame

from game_logic import Game2048, MoveResult, SpawnedTile


WIDTH, HEIGHT = 560, 820
BOARD_RECT = pygame.Rect(28, 230, 504, 504)
CELL_SIZE, GAP, PADDING = 109, 12, 16
BG_COLOR = (68, 166, 168)
HEADER_COLOR = (57, 195, 202)
CARD_COLOR = (122, 213, 215)
BOARD_COLOR = (215, 253, 251)
EMPTY_COLOR = (195, 239, 238)
SHADOW_COLOR = (43, 125, 129)
INK = (38, 113, 122)
WHITE = (250, 255, 255)
COLORS = {
    2: (143, 237, 235), 4: (0, 201, 224), 8: (0, 166, 234),
    16: (114, 152, 248), 32: (6, 122, 218), 64: (48, 79, 143),
    128: (101, 88, 192), 256: (140, 91, 199), 512: (193, 97, 178),
    1024: (233, 135, 118), 2048: (235, 181, 66),
}
# Slide then settle/pop; the highest speed skips both phases entirely.
SPEEDS = ((0.24, 0.15), (0.14, 0.10), (0.0, 0.0))
SPEED_NAMES = ("慢速", "标准", "极速")
DEFAULT_STORAGE = Path(__file__).resolve().parent / "data" / "preferences.json"


@dataclass
class Preferences:
    best_score: int = 0
    speed_index: int = 1

    @classmethod
    def load(cls, path):
        preferences = cls()
        if path is None:
            return preferences
        try:
            payload = Path(path).read_text(encoding="utf-8")
            data = json.loads(payload)
            if not isinstance(data, dict):
                return preferences
            best, speed = data.get("best_score"), data.get("speed_index")
            if type(best) is int and 0 <= best <= 10**15:
                preferences.best_score = best
            if type(speed) is int and speed in range(len(SPEEDS)):
                preferences.speed_index = speed
        except (OSError, ValueError):
            pass
        return preferences

    def save(self, path):
        if path is None:
            return
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps({
                "best_score": self.best_score,
                "speed_index": self.speed_index,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(path)
        except OSError:
            # A read-only directory must not prevent playing the game.
            pass


class FontBook:
    """Create each font/size once, including the sizes used by tile pops."""

    def __init__(self):
        directory = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        self.paths = {}
        for kind, names in {
            "ui": ("msyhbd.ttc", "msyh.ttc", "simhei.ttf"),
            "number": ("segoeuib.ttf", "arialbd.ttf"),
        }.items():
            self.paths[kind] = next(
                (str(directory / name) for name in names if (directory / name).is_file()),
                pygame.font.match_font(
                    "notosanscjk,wenquanyimicrohei,dejavusans,arial", bold=True
                ),
            )
        self.cache = {}

    def get(self, size, numeric=False):
        key = ("number" if numeric else "ui", max(8, round(size)))
        if key not in self.cache:
            self.cache[key] = pygame.font.Font(self.paths[key[0]], key[1])
        return self.cache[key]


@dataclass
class MoveAnimation:
    move: MoveResult
    spawn: SpawnedTile | None
    slide_duration: float
    settle_duration: float
    elapsed: float = 0.0

    @property
    def duration(self):
        return self.slide_duration + self.settle_duration


@dataclass(frozen=True)
class Button:
    rect: pygame.Rect
    label: str
    enabled: bool = True
    selected: bool = False
    danger: bool = False


class GameApp:
    def __init__(self, screen=None, *, storage_path=DEFAULT_STORAGE, seed=None):
        pygame.font.init()
        self.screen = screen
        self.canvas = pygame.Surface((WIDTH, HEIGHT))
        self.fonts = FontBook()
        self.storage_path = storage_path
        self.preferences = Preferences.load(storage_path)
        self.game = Game2048(seed=seed)
        self.undo_state = None  # Exactly one human move; not part of Game2048.step().
        self.animation = None
        self.pending_action = None  # Keep at most the latest input during a slide.
        self.view = "game"
        self.running = True
        self.pointer = (-1, -1)
        self.popup_reward = 0
        self.popup_age = 1.0
        self.goal_age = 10.0

    def finish_animation(self):
        self.animation = None
        self.pending_action = None

    def play_human(self, action):
        if self.view != "game" or self.game.game_over:
            return False
        if self.animation is not None:
            self.pending_action = action
            return False
        before = self.game.get_state()
        reached_goal = bool((before.board >= 2048).any())
        _, reward, _ = self.game.step(action, capture_animation=True)
        if self.game.last_move is None:
            return False
        self.undo_state = before
        slide, settle = SPEEDS[self.preferences.speed_index]
        if slide + settle > 0:
            self.animation = MoveAnimation(
                self.game.last_move, self.game.last_spawn, slide, settle
            )
        if reward:
            self.popup_reward, self.popup_age = reward, 0.0
        if not reached_goal and (self.game.board >= 2048).any():
            self.goal_age = 0.0
        if self.game.score > self.preferences.best_score:
            self.preferences.best_score = self.game.score
            self.preferences.save(self.storage_path)
        return True

    def undo(self):
        if self.undo_state is None:
            return False
        self.finish_animation()
        self.game.set_state(self.undo_state)
        self.undo_state = None
        self.popup_age, self.goal_age = 1.0, 10.0
        self.view = "game"
        return True

    def new_game(self):
        self.finish_animation()
        self.game.reset()
        self.undo_state = None
        self.popup_age, self.goal_age = 1.0, 10.0
        self.view = "game"
        self.preferences.save(self.storage_path)

    def set_speed(self, index):
        if type(index) is not int or index not in range(len(SPEEDS)):
            raise ValueError("speed index must be 0, 1 or 2")
        self.finish_animation()
        self.preferences.speed_index = index
        self.preferences.save(self.storage_path)

    def update(self, seconds):
        self.popup_age += seconds
        self.goal_age += seconds
        if self.animation is not None:
            self.animation.elapsed += seconds
            if self.animation.elapsed >= self.animation.duration:
                queued = self.pending_action
                self.finish_animation()
                if queued is not None:
                    self.play_human(queued)

    def buttons(self):
        items = {
            "undo": Button(pygame.Rect(280, 158, 112, 56), "撤回",
                           self.undo_state is not None and self.view == "game"),
            "menu": Button(pygame.Rect(404, 158, 112, 56),
                           "菜单" if self.view == "game" else "返回"),
        }
        if self.view == "menu":
            for index, label in enumerate(SPEED_NAMES):
                items[f"speed_{index}"] = Button(
                    pygame.Rect(72 + index * 140, 370, 132, 52), label,
                    selected=self.preferences.speed_index == index,
                )
            items.update({
                "about": Button(pygame.Rect(72, 466, 200, 58), "关于"),
                "reset": Button(pygame.Rect(288, 466, 200, 58), "重新开局"),
                "back": Button(pygame.Rect(72, 550, 200, 58), "返回游戏"),
                "exit": Button(pygame.Rect(288, 550, 200, 58), "退出游戏", danger=True),
            })
        elif self.view == "about":
            items["about_back"] = Button(pygame.Rect(138, 618, 284, 56), "返回菜单")
        elif self.game.game_over and self.animation is None:
            items.update({
                "over_undo": Button(pygame.Rect(82, 500, 184, 56), "撤回一步",
                                    self.undo_state is not None),
                "over_reset": Button(pygame.Rect(286, 500, 192, 56), "重新开局"),
            })
        return items

    def activate(self, name):
        button = self.buttons().get(name)
        if button is None or not button.enabled:
            return
        if name in ("undo", "over_undo"):
            self.undo()
        elif name in ("reset", "over_reset"):
            self.new_game()
        elif name == "menu":
            self.finish_animation()
            self.view = "game" if self.view == "menu" else "menu"
        elif name == "about":
            self.view = "about"
        elif name == "about_back":
            self.view = "menu"
        elif name == "back":
            self.view = "game"
        elif name == "exit":
            self.preferences.save(self.storage_path)
            self.running = False
        elif name.startswith("speed_"):
            self.set_speed(int(name[-1]))

    def window_to_canvas(self, position):
        if self.screen is None:
            return position
        width, height = self.screen.get_size()
        scale = min(width / WIDTH, height / HEIGHT)
        offset = ((width - WIDTH * scale) / 2, (height - HEIGHT * scale) / 2)
        return ((position[0] - offset[0]) / scale, (position[1] - offset[1]) / scale)

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.VIDEORESIZE and self.screen is not None:
            self.screen = pygame.display.set_mode(
                (max(380, event.w), max(560, event.h)), pygame.RESIZABLE
            )
        elif event.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN):
            self.pointer = self.window_to_canvas(event.pos)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for name, button in self.buttons().items():
                    if button.rect.collidepoint(self.pointer):
                        self.activate(name)
                        break
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.finish_animation()
                self.view = {"game": "menu", "menu": "game", "about": "menu"}[self.view]
                return
            if self.view == "menu":
                if event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    self.set_speed(event.key - pygame.K_1)
                return
            if self.view != "game":
                return
            if event.key in (pygame.K_u, pygame.K_BACKSPACE) or (
                event.key == pygame.K_z and getattr(event, "mod", 0) & pygame.KMOD_CTRL
            ):
                self.undo()
            elif event.key == pygame.K_r:
                self.new_game()
            else:
                action = {
                    pygame.K_UP: 0, pygame.K_w: 0,
                    pygame.K_DOWN: 1, pygame.K_s: 1,
                    pygame.K_LEFT: 2, pygame.K_a: 2,
                    pygame.K_RIGHT: 3, pygame.K_d: 3,
                }.get(event.key)
                if action is not None:
                    self.play_human(action)

    def text(self, text, size, center, color=WHITE, *, width=None, numeric=False,
             shadow=False):
        font = self.fonts.get(size, numeric)
        while width is not None and font.size(text)[0] > width and size > 8:
            size -= 1
            font = self.fonts.get(size, numeric)
        image = font.render(text, True, color)
        rect = image.get_rect(center=(round(center[0]), round(center[1])))
        if shadow:
            shade = font.render(text, True, (48, 135, 151))
            self.canvas.blit(shade, rect.move(0, max(1, round(size / 18))))
        self.canvas.blit(image, rect)

    def panel(self, rect, color, *, shadow=True, radius=3):
        if shadow:
            pygame.draw.rect(self.canvas, SHADOW_COLOR, rect.move(6, 7), border_radius=radius)
        pygame.draw.rect(self.canvas, color, rect, border_radius=radius)

    def draw_button(self, button, *, small=False):
        color = CARD_COLOR
        foreground = WHITE
        if button.danger:
            color = (88, 177, 184)
        if button.selected:
            color = (24, 158, 179)
        if not button.enabled:
            color, foreground = (105, 199, 202), (171, 229, 229)
        elif button.rect.collidepoint(self.pointer):
            color = tuple(min(255, channel + 12) for channel in color)
        self.panel(button.rect, color, shadow=button.enabled)
        if button.selected:
            pygame.draw.rect(self.canvas, WHITE,
                             (button.rect.x + 14, button.rect.bottom - 7, button.rect.w - 28, 3))
        self.text(button.label, 23 if small else 24, button.rect.center,
                  foreground, width=button.rect.w - 20)

    @staticmethod
    def cell_center(position):
        row, col = position
        return (BOARD_RECT.x + PADDING + col * (CELL_SIZE + GAP) + CELL_SIZE // 2,
                BOARD_RECT.y + PADDING + row * (CELL_SIZE + GAP) + CELL_SIZE // 2)

    def draw_tile(self, value, center, scale=1.0):
        if not value or scale <= 0.01:
            return
        size = max(1, round(CELL_SIZE * scale))
        rect = pygame.Rect(0, 0, size, size)
        rect.center = (round(center[0]), round(center[1]))
        color = COLORS.get(value, (33, 105, 128))
        self.panel(rect, color, radius=2)
        digits = len(str(value))
        base_size = 56 if digits <= 2 else (45 if digits == 3 else (36 if digits == 4 else 29))
        self.text(str(value), max(8, round(base_size * scale)), rect.center,
                  numeric=True, width=size - max(4, round(12 * scale)), shadow=True)

    def draw_header(self):
        self.text("2048", 34, (80, 37), numeric=True)
        self.text("每一次合并，都更进一步", 14, (416, 39), width=230)
        self.panel(pygame.Rect(28, 76, 504, 658), HEADER_COLOR)
        pygame.draw.rect(self.canvas, CARD_COLOR, (44, 92, 220, 122))
        self.text("当前分数", 14, (154, 111))
        self.text(str(self.game.score), 62, (154, 159), numeric=True, width=196)
        pygame.draw.rect(self.canvas, CARD_COLOR, (280, 92, 236, 54))
        self.text("最高", 15, (312, 119))
        self.text(str(self.preferences.best_score), 28, (421, 118), numeric=True, width=166)
        items = self.buttons()
        self.draw_button(items["undo"])
        self.draw_button(items["menu"])
        if self.popup_age < 0.75 and self.popup_reward:
            self.text(f"+{self.popup_reward}", 17,
                      (235, 201 - self.popup_age * 20), width=54, numeric=True)

    def draw_board(self):
        pygame.draw.rect(self.canvas, BOARD_COLOR, BOARD_RECT)
        for row in range(4):
            for col in range(4):
                rect = pygame.Rect(0, 0, CELL_SIZE, CELL_SIZE)
                rect.center = self.cell_center((row, col))
                pygame.draw.rect(self.canvas, EMPTY_COLOR, rect, border_radius=2)
        animation = self.animation
        if animation is None:
            for row in range(4):
                for col in range(4):
                    self.draw_tile(int(self.game.board[row, col]), self.cell_center((row, col)))
            return
        if animation.elapsed < animation.slide_duration:
            t = animation.elapsed / animation.slide_duration
            eased = 1 - (1 - t) ** 3
            for motion in animation.move.motions:
                source, target = self.cell_center(motion.source), self.cell_center(motion.destination)
                center = (source[0] + (target[0] - source[0]) * eased,
                          source[1] + (target[1] - source[1]) * eased)
                self.draw_tile(motion.value, center)
        else:
            t = min(1.0, (animation.elapsed - animation.slide_duration) / animation.settle_duration)
            for row in range(4):
                for col in range(4):
                    scale = 1.0
                    if (row, col) in animation.move.merged_cells:
                        scale += 0.12 * math.sin(math.pi * t)
                    self.draw_tile(int(animation.move.board[row, col]),
                                   self.cell_center((row, col)), scale)
            if animation.spawn is not None:
                self.draw_tile(animation.spawn.value, self.cell_center(animation.spawn.position),
                               1 - (1 - t) ** 3)

    def draw_modal_background(self):
        overlay = pygame.Surface(BOARD_RECT.size, pygame.SRCALPHA)
        overlay.fill((215, 253, 251, 220))
        self.canvas.blit(overlay, BOARD_RECT.topleft)
        self.panel(pygame.Rect(48, 254, 464, 450), BOARD_COLOR)

    def draw_menu(self):
        self.draw_modal_background()
        self.text("游戏菜单", 27, (280, 291), INK)
        self.text("动画速度", 17, (121, 340), INK)
        self.text("极速档无过渡动画", 13, (411, 340), INK)
        for name, button in self.buttons().items():
            if name not in ("undo", "menu"):
                self.draw_button(button, small=True)
        self.text("按 1 / 2 / 3 切换速度  ·  Esc 返回", 13, (280, 656), INK, width=420)

    def draw_about(self):
        self.draw_modal_background()
        self.text("2048", 68, (280, 327), INK, numeric=True)
        self.text("把小小的数字，合成更大的可能。", 16, (280, 392), INK, width=410)
        for text, y in [
            ("滑动棋盘，让相同数字相遇并合并。", 443),
            ("达到 2048 后，也可以继续挑战。", 476),
            ("方向键 / WASD    移动", 528),
            ("U / Ctrl+Z    撤回一步     R    重新开局", 560),
        ]:
            self.text(text, 15, (280, y), INK, width=406)
        self.draw_button(self.buttons()["about_back"], small=True)

    def draw_game_over(self):
        overlay = pygame.Surface(BOARD_RECT.size, pygame.SRCALPHA)
        overlay.fill((215, 253, 251, 215))
        self.canvas.blit(overlay, BOARD_RECT.topleft)
        self.text("本局结束", 38, (280, 410), INK)
        self.text(f"本局得分  {self.game.score}", 20, (280, 461), INK, width=420)
        for name in ("over_undo", "over_reset"):
            self.draw_button(self.buttons()[name], small=True)
        self.text("撤回一步，再试一个方向。", 15, (280, 599), INK)

    def render(self):
        self.canvas.fill(BG_COLOR)
        self.draw_header()
        self.draw_board()
        if self.view == "menu":
            self.draw_menu()
        elif self.view == "about":
            self.draw_about()
        elif self.game.game_over and self.animation is None:
            self.draw_game_over()
        if self.goal_age < 3.5 and self.view == "game":
            self.text("已合成 2048！继续挑战更大的数字吧。", 16, (280, 770), width=500)
        else:
            self.text("方向键 / WASD 移动  ·  U 撤回  ·  Esc 菜单", 14, (280, 770), width=500)
        self.text("合并相同数字，向 2048 前进", 12, (280, 799), (192, 240, 238))
        if self.screen is not None:
            width, height = self.screen.get_size()
            scale = min(width / WIDTH, height / HEIGHT)
            target = (round(WIDTH * scale), round(HEIGHT * scale))
            self.screen.fill(BG_COLOR)
            surface = self.canvas if target == (WIDTH, HEIGHT) else pygame.transform.smoothscale(
                self.canvas, target
            )
            self.screen.blit(surface, ((width - target[0]) // 2, (height - target[1]) // 2))
        return self.canvas


def main():
    parser = argparse.ArgumentParser(description="2048")
    parser.add_argument("--seed", type=int, default=None, help="Optional reproducible game seed")
    args = parser.parse_args()
    pygame.display.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("2048")
    clock = pygame.time.Clock()
    app = GameApp(screen, seed=args.seed)
    try:
        while app.running:
            elapsed = clock.tick(60) / 1000.0
            for event in pygame.event.get():
                app.handle_event(event)
            app.update(elapsed)
            app.render()
            pygame.display.flip()
    finally:
        app.preferences.save(app.storage_path)
        pygame.quit()


if __name__ == "__main__":
    main()
