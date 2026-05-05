import random
import time
from pathlib import Path

try:
    import pygame
except ImportError:
    pygame = None

from algorithms.ga import run_ga
from algorithms.hybrid import run_hybrid
from algorithms.pso import run_pso
from core.fitness import calculate_fitness
from simulation.traffic_simulation import TrafficSimulation


WIDTH = 1280
HEIGHT = 820
FPS = 60

INTERSECTIONS = 6
ROAD_W = 124
CAR_W = 34
CAR_H = 62
CYCLE_LENGTH = 60
YELLOW_TIME = 5
SIGNAL_SPEED = 0.18

MAP_LEFT = 24
MAP_RIGHT = 900
SIDEBAR_X = 928
SIDEBAR_W = 328
COL_X = [170, 435, 700]
ROW_Y = [278, 552]
BASELINE_TIMINGS = [30, 30, 30, 30, 30, 30]
ASSET_DIR = Path(__file__).parent / "assets"
SCENARIOS = {
    "Low": 0.04,
    "Medium": 0.08,
    "High": 0.13,
}


COLORS = {
    "background": (249, 250, 251),
    "grass": (231, 245, 235),
    "grass_dark": (199, 224, 207),
    "grass_light": (217, 239, 223),
    "road": (99, 106, 116),
    "road_dark": (75, 85, 99),
    "curb": (209, 213, 219),
    "sidewalk": (229, 231, 235),
    "lane": (242, 244, 247),
    "yellow": (245, 158, 11),
    "panel": (255, 255, 255),
    "text": (31, 41, 55),
    "muted": (107, 114, 128),
    "blue": (37, 99, 235),
    "red": (239, 68, 68),
    "green": (34, 197, 94),
    "amber": (245, 158, 11),
    "black": (17, 24, 39),
    "white": (255, 255, 255),
    "sky": (239, 246, 255),
    "ui_orange": (245, 158, 11),
    "border": (229, 231, 235),
}


class FitnessIndividual:
    def __init__(self, genome):
        self.genome = genome
        self.fitness = None


class Car:
    def __init__(self, column, direction, image):
        self.column = column
        self.direction = direction
        self.x = lane_x(column, direction)
        self.y = -CAR_H - random.randint(0, 180) if direction == 1 else HEIGHT + random.randint(0, 180)
        self.speed = random.uniform(1.8, 2.8)
        self.target_row = 0 if direction == 1 else len(ROW_Y) - 1
        self.wait_time = 0
        self.stops = 0
        self.was_stopped = False
        self.done = False
        self.image = image

    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), CAR_W, CAR_H)


class TrafficGame:
    def __init__(self, timings, seed=7, density=0.055):
        self.rng = random.Random(seed)
        self.seed = seed
        self.density = density
        self.timings = timings[:]
        self.cars = []
        self.step = 0
        self.spawn_meter = 0
        self.completed = 0
        self.total_wait = 0
        self.total_stops = 0
        self.car_images = {}
        self.mode_name = "Baseline fixed 30s"

    def set_images(self, car_images):
        self.car_images = car_images

    def reset(self, timings=None, mode_name=None):
        if timings is not None:
            self.timings = timings[:]
        if mode_name:
            self.mode_name = mode_name
        self.cars = []
        self.step = 0
        self.spawn_meter = 0
        self.completed = 0
        self.total_wait = 0
        self.total_stops = 0

    def set_density(self, density):
        self.density = density

    def signal_index(self, column, row):
        return row * 3 + column

    def light_state(self, column, row):
        index = self.signal_index(column, row)
        signal_time = (self.step * SIGNAL_SPEED) % CYCLE_LENGTH
        green = self.timings[index]
        if signal_time < green:
            return "green"
        if signal_time < green + YELLOW_TIME:
            return "yellow"
        return "red"

    def spawn_car(self):
        column = self.rng.randrange(3)
        direction = self.rng.choice([1, -1])
        key = self.rng.choice(list(self.car_images.keys()))
        image = self.car_images[key][direction]
        self.cars.append(Car(column, direction, image))

    def update(self):
        self.step += 1
        self.spawn_meter += self.density
        while self.spawn_meter >= 1:
            self.spawn_meter -= 1
            self.spawn_car()

        ordered = sorted(self.cars, key=lambda car: car.y, reverse=True)
        for car in ordered:
            if car.done:
                continue

            stopped = self.should_stop(car)
            front = self.nearest_front_car(car)
            if front and self.close_to_front_car(car, front):
                stopped = True

            if stopped:
                car.wait_time += 1
                self.total_wait += 1
                if not car.was_stopped:
                    car.stops += 1
                    self.total_stops += 1
                car.was_stopped = True
            else:
                car.was_stopped = False
                car.y += car.direction * car.speed
                self.update_target(car)

            if car.y > HEIGHT + CAR_H or car.y < -CAR_H * 2:
                car.done = True
                self.completed += 1

        self.cars = [car for car in self.cars if not car.done]

    def should_stop(self, car):
        if car.target_row < 0 or car.target_row >= len(ROW_Y):
            return False

        row_y = ROW_Y[car.target_row]
        if car.direction == 1:
            approaching = car.y + CAR_H >= row_y - 56 and car.y + CAR_H < row_y + 12
        else:
            approaching = car.y <= row_y + 56 and car.y > row_y - 12

        if not approaching:
            return False

        state = self.light_state(car.column, car.target_row)
        if state == "red":
            return True
        if state == "yellow" and self.rng.random() < 0.55:
            return True
        return False

    def update_target(self, car):
        if car.direction == 1 and car.target_row < len(ROW_Y):
            if car.y > ROW_Y[car.target_row] + 36:
                car.target_row += 1
        elif car.direction == -1 and car.target_row >= 0:
            if car.y + CAR_H < ROW_Y[car.target_row] - 36:
                car.target_row -= 1

    def nearest_front_car(self, car):
        candidates = []
        for other in self.cars:
            if other is car or other.column != car.column or other.direction != car.direction:
                continue
            same_lane = abs(other.x - car.x) < 5
            in_front = other.y > car.y if car.direction == 1 else other.y < car.y
            if same_lane and in_front:
                candidates.append(other)
        if not candidates:
            return None
        return min(candidates, key=lambda other: abs(other.y - car.y))

    def close_to_front_car(self, car, front):
        distance = abs(front.y - car.y)
        return 0 < distance < CAR_H + 14

    def queue_lengths(self):
        queues = [0] * INTERSECTIONS
        for car in self.cars:
            if car.was_stopped and 0 <= car.target_row < len(ROW_Y):
                queues[self.signal_index(car.column, car.target_row)] += 1
        return queues

    def metrics(self):
        active = len(self.cars)
        queue = sum(self.queue_lengths())
        avg_wait = self.total_wait / max(1, active + self.completed)
        live_fitness = avg_wait + 2 * queue
        return {
            "active": active,
            "completed": self.completed,
            "avg_wait": avg_wait,
            "queue": queue,
            "stops": self.total_stops,
            "fitness": live_fitness,
        }


def lane_x(column, direction):
    center = COL_X[column]
    return center - 42 if direction == 1 else center + 10


def draw_text(surface, font, text, x, y, color=COLORS["text"], center=False):
    rendered = font.render(text, True, color)
    rect = rendered.get_rect()
    if center:
        rect.center = (x, y)
    else:
        rect.topleft = (x, y)
    surface.blit(rendered, rect)


def fit_text(font, text, max_width):
    if font.size(text)[0] <= max_width:
        return text
    suffix = "..."
    trimmed = text
    while trimmed and font.size(trimmed + suffix)[0] > max_width:
        trimmed = trimmed[:-1]
    return trimmed + suffix


def draw_panel(surface, rect):
    shadow = pygame.Surface((rect.w + 12, rect.h + 12), pygame.SRCALPHA)
    pygame.draw.rect(shadow, (15, 23, 42, 18), (6, 7, rect.w, rect.h), border_radius=18)
    surface.blit(shadow, (rect.x - 6, rect.y - 6))
    pygame.draw.rect(surface, COLORS["panel"], rect, border_radius=18)
    pygame.draw.rect(surface, (238, 242, 247), rect, 1, border_radius=18)


def create_car_asset(path, body_color):
    surface = pygame.Surface((CAR_W + 6, CAR_H + 4), pygame.SRCALPHA)
    shade = tuple(max(0, channel - 45) for channel in body_color)
    pygame.draw.ellipse(surface, (0, 0, 0, 42), (3, CAR_H - 5, CAR_W, 8))
    pygame.draw.rect(surface, shade, (7, 10, 25, 47), border_radius=9)
    pygame.draw.rect(surface, body_color, (4, 4, 31, 51), border_radius=11)
    pygame.draw.rect(surface, (255, 255, 255, 90), (7, 8, 25, 4), border_radius=2)
    pygame.draw.polygon(surface, (191, 219, 254), [(8, 15), (31, 15), (27, 29), (12, 29)])
    pygame.draw.polygon(surface, (219, 234, 254), [(10, 36), (27, 36), (31, 48), (7, 48)])
    pygame.draw.rect(surface, shade, (7, 30, 25, 4), border_radius=2)
    pygame.draw.circle(surface, COLORS["black"], (4, 17), 4)
    pygame.draw.circle(surface, COLORS["black"], (35, 17), 4)
    pygame.draw.circle(surface, COLORS["black"], (4, 47), 4)
    pygame.draw.circle(surface, COLORS["black"], (35, 47), 4)
    pygame.draw.circle(surface, (254, 240, 138), (12, 6), 3)
    pygame.draw.circle(surface, (254, 240, 138), (27, 6), 3)
    pygame.image.save(surface, path)


def create_asphalt_asset(path):
    rng = random.Random(14)
    tile = pygame.Surface((96, 96))
    tile.fill(COLORS["road"])
    for _ in range(500):
        shade = rng.randint(-10, 10)
        base = max(72, min(124, 104 + shade))
        color = (base, base + rng.randint(-1, 3), base + rng.randint(-1, 4))
        x = rng.randrange(96)
        y = rng.randrange(96)
        tile.set_at((x, y), color)
    for _ in range(10):
        x = rng.randrange(96)
        y = rng.randrange(96)
        pygame.draw.line(tile, (90, 96, 103), (x, y), (x + rng.randint(8, 24), y + rng.randint(-2, 2)), 1)
    pygame.image.save(tile, path)


def load_environment_assets():
    ASSET_DIR.mkdir(exist_ok=True)
    asphalt_path = ASSET_DIR / "dashboard_asphalt_tile.png"
    if not asphalt_path.exists():
        create_asphalt_asset(asphalt_path)
    asphalt = pygame.image.load(str(asphalt_path))
    if pygame.display.get_surface() is not None:
        asphalt = asphalt.convert()
    return {"asphalt": asphalt}


def load_car_images():
    ASSET_DIR.mkdir(exist_ok=True)
    custom_paths = sorted(ASSET_DIR.glob("custom_car_*.png"))
    if custom_paths:
        images = {}
        for index, path in enumerate(custom_paths):
            down = pygame.image.load(str(path))
            down = pygame.transform.smoothscale(down, (CAR_W + 6, CAR_H + 4))
            if pygame.display.get_surface() is not None:
                down = down.convert_alpha()
            up = pygame.transform.rotate(down, 180)
            images[f"custom_{index}"] = {1: down, -1: up}
        return images

    specs = {
        "red": (220, 38, 38),
        "orange": (249, 115, 22),
        "white": (241, 245, 249),
        "blue": (37, 99, 235),
    }

    images = {}
    for name, color in specs.items():
        path = ASSET_DIR / f"realistic_top_car_{name}.png"
        if not path.exists():
            create_car_asset(path, color)
        down = pygame.image.load(str(path))
        if pygame.display.get_surface() is not None:
            down = down.convert_alpha()
        up = pygame.transform.rotate(down, 180)
        images[name] = {1: down, -1: up}
    return images


def blit_tiled(surface, tile, rect):
    old_clip = surface.get_clip()
    surface.set_clip(rect)
    for x in range(rect.left, rect.right, tile.get_width()):
        for y in range(rect.top, rect.bottom, tile.get_height()):
            surface.blit(tile, (x, y))
    surface.set_clip(old_clip)


def draw_road_rect(surface, asphalt, rect):
    pygame.draw.rect(surface, COLORS["curb"], rect.inflate(12, 12), border_radius=12)
    pygame.draw.rect(surface, COLORS["sidewalk"], rect.inflate(5, 5), border_radius=9)
    blit_tiled(surface, asphalt, rect)
    pygame.draw.rect(surface, COLORS["road_dark"], rect, 1, border_radius=6)


def draw_crosswalk(surface, x, y):
    stripe_color = (235, 239, 244)
    for offset in range(-48, 49, 16):
        pygame.draw.rect(surface, stripe_color, (x - 62, y + offset - 4, 42, 8), border_radius=1)
        pygame.draw.rect(surface, stripe_color, (x + 20, y + offset - 4, 42, 8), border_radius=1)
        pygame.draw.rect(surface, stripe_color, (x + offset - 4, y - 62, 8, 42), border_radius=1)
        pygame.draw.rect(surface, stripe_color, (x + offset - 4, y + 20, 8, 42), border_radius=1)


def draw_city_details(surface):
    pygame.draw.rect(surface, COLORS["sky"], (MAP_LEFT, 24, MAP_RIGHT - MAP_LEFT, 100), border_radius=18)
    for x in range(58, MAP_RIGHT - 38, 118):
        pygame.draw.circle(surface, COLORS["grass_light"], (x, 182), 13)
        pygame.draw.circle(surface, COLORS["grass_dark"], (x + 7, 179), 8)
        pygame.draw.rect(surface, (148, 116, 80), (x - 2, 193, 4, 12), border_radius=2)

    for x in range(62, MAP_RIGHT - 80, 164):
        pygame.draw.rect(surface, (226, 232, 240), (x, 624, 82, 46), border_radius=8)
        pygame.draw.rect(surface, (203, 213, 225), (x + 10, 636, 62, 4), border_radius=2)
        pygame.draw.rect(surface, (203, 213, 225), (x + 10, 650, 42, 4), border_radius=2)


def draw_roads(surface, environment):
    surface.fill(COLORS["background"])
    pygame.draw.rect(surface, COLORS["grass"], (MAP_LEFT, 126, MAP_RIGHT - MAP_LEFT, HEIGHT - 260), border_radius=22)
    asphalt = environment["asphalt"]
    draw_city_details(surface)

    for x in COL_X:
        rect = pygame.Rect(x - ROAD_W // 2, 80, ROAD_W, HEIGHT - 210)
        draw_road_rect(surface, asphalt, rect)
        pygame.draw.line(surface, COLORS["yellow"], (x, 96), (x, HEIGHT - 148), 3)
        for y in range(100, HEIGHT - 160, 70):
            pygame.draw.line(surface, COLORS["lane"], (x - 34, y), (x - 34, y + 36), 3)
            pygame.draw.line(surface, COLORS["lane"], (x + 34, y), (x + 34, y + 36), 3)

    for y in ROW_Y:
        rect = pygame.Rect(70, y - ROAD_W // 2, MAP_RIGHT - 118, ROAD_W)
        draw_road_rect(surface, asphalt, rect)
        pygame.draw.line(surface, COLORS["yellow"], (84, y), (MAP_RIGHT - 64, y), 3)
        for x in range(95, MAP_RIGHT - 90, 80):
            pygame.draw.line(surface, COLORS["lane"], (x, y - 34), (x + 40, y - 34), 3)
            pygame.draw.line(surface, COLORS["lane"], (x, y + 34), (x + 40, y + 34), 3)

    for row, y in enumerate(ROW_Y):
        for column, x in enumerate(COL_X):
            intersection = pygame.Rect(x - 70, y - 70, 140, 140)
            blit_tiled(surface, asphalt, intersection)
            pygame.draw.rect(surface, (88, 93, 99), intersection, 2, border_radius=4)
            draw_crosswalk(surface, x, y)


def draw_signals(surface, game, small_font):
    for row, y in enumerate(ROW_Y):
        for column, x in enumerate(COL_X):
            state = game.light_state(column, row)
            timing = game.timings[game.signal_index(column, row)]
            box = pygame.Rect(x + 48, y - 58, 26, 58)
            pygame.draw.rect(surface, (17, 24, 39), box, border_radius=9)
            inactive = (55, 65, 81)
            light_specs = [
                ("red", COLORS["red"], y - 45),
                ("yellow", COLORS["amber"], y - 29),
                ("green", COLORS["green"], y - 13),
            ]
            for name, color, cy in light_specs:
                active = state == name
                if active:
                    glow = pygame.Surface((26, 26), pygame.SRCALPHA)
                    pygame.draw.circle(glow, (*color, 72), (13, 13), 12)
                    surface.blit(glow, (x + 48, cy - 13))
                pygame.draw.circle(surface, color if active else inactive, (x + 61, cy), 6)
                pygame.draw.circle(surface, (255, 255, 255, 54), (x + 59, cy - 2), 2)
            draw_text(surface, small_font, f"I{row * 3 + column + 1}", x - 38, y - 58, COLORS["white"])
            draw_text(surface, small_font, f"G={int(round(timing))}", x - 38, y - 38, COLORS["white"])


def draw_cars(surface, game):
    for car in game.cars:
        shadow = pygame.Surface((CAR_W + 8, CAR_H + 8), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (0, 0, 0, 70), (2, CAR_H - 10, CAR_W + 4, 14))
        surface.blit(shadow, (int(car.x) - 4, int(car.y) + 2))
        surface.blit(car.image, (int(car.x), int(car.y)))


def draw_hud(surface, game, font, small_font, baseline_fitness, optimized_fitness, optimized_solution, message):
    metrics = game.metrics()
    draw_panel(surface, pygame.Rect(24, 20, WIDTH - 48, 88))
    draw_text(surface, font, "Traffic Signal Optimization System", 46, 36)
    draw_text(surface, small_font, "AI-powered timing dashboard for a six-intersection traffic network", 48, 68, COLORS["muted"])
    status_color = COLORS["green"] if len(game.cars) else COLORS["muted"]
    pygame.draw.circle(surface, status_color, (914, 47), 6)
    draw_text(surface, small_font, f"Mode: {game.mode_name}", 932, 36, COLORS["text"])
    timings = " ".join(str(int(round(value))) for value in optimized_solution)
    draw_text(surface, small_font, f"Signal plan: {timings}", 932, 66, COLORS["muted"])

    improvement = ((baseline_fitness - optimized_fitness) / baseline_fitness * 100) if baseline_fitness else 0
    return metrics, improvement


def draw_stat_card(surface, small_font, title, value, rect, accent):
    draw_panel(surface, rect)
    pygame.draw.rect(surface, accent, (rect.x + 14, rect.y + 14, 4, rect.h - 28), border_radius=3)
    draw_text(surface, small_font, title, rect.x + 28, rect.y + 14, COLORS["muted"])
    draw_text(surface, small_font, value, rect.x + 28, rect.y + 42, COLORS["text"])


def draw_bottom_stats(surface, small_font, metrics, improvement):
    stats = [
        ("Active Cars", str(metrics["active"]), COLORS["blue"]),
        ("Avg Wait", f"{metrics['avg_wait']:.2f}", COLORS["amber"]),
        ("Queue", str(metrics["queue"]), COLORS["red"]),
        ("Stops", str(metrics["stops"]), COLORS["muted"]),
        ("Improvement", f"{improvement:.1f}%", COLORS["green"] if improvement >= 0 else COLORS["red"]),
    ]
    card_w = 232
    for index, (title, value, color) in enumerate(stats):
        rect = pygame.Rect(36 + index * (card_w + 12), HEIGHT - 94, card_w, 62)
        draw_stat_card(surface, small_font, title, value, rect, color)


def draw_buttons(surface, font, buttons, mouse_pos):
    for label, rect in buttons.items():
        primary = label == "Optimize"
        danger = label == "Stop"
        success = label == "Start"
        hovered = rect.collidepoint(mouse_pos)
        if primary:
            fill = COLORS["blue"] if not hovered else (59, 130, 246)
            text_color = COLORS["white"]
            border = COLORS["blue"]
        elif success:
            fill = COLORS["green"] if not hovered else (74, 222, 128)
            text_color = COLORS["white"]
            border = COLORS["green"]
        elif danger:
            fill = (254, 242, 242) if not hovered else (254, 226, 226)
            text_color = COLORS["red"]
            border = (252, 165, 165)
        else:
            fill = COLORS["white"] if not hovered else (239, 246, 255)
            text_color = COLORS["blue"]
            border = (147, 197, 253)

        pygame.draw.rect(surface, (15, 23, 42, 16), rect.move(2, 3), border_radius=20)
        pygame.draw.rect(surface, fill, rect, border_radius=20)
        pygame.draw.rect(surface, border, rect, 1, border_radius=20)
        draw_text(surface, font, label, rect.centerx, rect.centery, text_color, center=True)


def draw_status_message(surface, small_font, message):
    rect = pygame.Rect(SIDEBAR_X, 524, SIDEBAR_W, 26)
    pygame.draw.rect(surface, (239, 246, 255), rect, border_radius=13)
    pygame.draw.circle(surface, COLORS["blue"], (rect.x + 14, rect.centery), 4)
    text = fit_text(small_font, message, rect.w - 42)
    draw_text(surface, small_font, text, rect.x + 28, rect.y + 4, COLORS["muted"])


def draw_pill_group(surface, small_font, options, selected_value):
    if options:
        xs = [rect.x for rect in options.values()]
        ys = [rect.y for rect in options.values()]
        rights = [rect.right for rect in options.values()]
        bottoms = [rect.bottom for rect in options.values()]
        outer = pygame.Rect(min(xs), min(ys), max(rights) - min(xs), max(bottoms) - min(ys))
        pygame.draw.rect(surface, (243, 244, 246), outer.inflate(4, 4), border_radius=22)
    for name, rect in options.items():
        selected = name == selected_value
        fill = COLORS["blue"] if selected else (249, 250, 251)
        border = COLORS["blue"] if selected else (229, 231, 235)
        text_color = COLORS["white"] if selected else COLORS["text"]
        pygame.draw.rect(surface, fill, rect, border_radius=18)
        pygame.draw.rect(surface, border, rect, 1, border_radius=18)
        draw_text(surface, small_font, name, rect.centerx, rect.centery, text_color, center=True)


def draw_algorithm_menu(surface, font, small_font, selected_algorithm, algorithm_rects, selected_scenario, scenario_rects):
    panel = pygame.Rect(SIDEBAR_X, 126, SIDEBAR_W, 266)
    draw_panel(surface, panel)
    draw_text(surface, font, "Control Panel", panel.x + 18, panel.y + 16)

    draw_text(surface, small_font, "Algorithm", panel.x + 18, panel.y + 60, COLORS["muted"])
    draw_pill_group(surface, small_font, algorithm_rects, selected_algorithm)

    draw_text(surface, small_font, "Traffic level", panel.x + 172, panel.y + 60, COLORS["muted"])
    draw_pill_group(surface, small_font, scenario_rects, selected_scenario)


def draw_insight_panel(surface, font, small_font, baseline_fitness, optimized_fitness, best_algorithm, selected_scenario):
    panel = pygame.Rect(SIDEBAR_X, 612, 252, 156)
    draw_panel(surface, panel)
    improvement = ((baseline_fitness - optimized_fitness) / baseline_fitness * 100) if baseline_fitness else 0
    title = "Insight"
    draw_text(surface, font, title, panel.x + 18, panel.y + 16)
    draw_text(surface, small_font, f"Scenario: {selected_scenario} traffic", panel.x + 18, panel.y + 52, COLORS["muted"])
    draw_text(surface, small_font, f"{best_algorithm} reduced fitness by", panel.x + 18, panel.y + 82, COLORS["text"])
    draw_text(surface, font, f"{improvement:.1f}%", panel.x + 18, panel.y + 106, COLORS["green"] if improvement >= 0 else COLORS["red"])
    draw_text(surface, small_font, "Best timing uses the lowest", panel.x + 118, panel.y + 110, COLORS["muted"])
    draw_text(surface, small_font, "wait + congestion cost.", panel.x + 118, panel.y + 132, COLORS["muted"])


def compare_algorithms():
    results = []
    for name in ["PSO", "GA", "Hybrid"]:
        solution, fitness, elapsed = run_optimizer(name)
        results.append(
            {
                "name": name,
                "fitness": fitness,
                "elapsed": elapsed,
                "timings": [int(round(value)) for value in solution],
            }
        )
    return sorted(results, key=lambda item: item["fitness"])


def draw_comparison_panel(surface, font, small_font, results):
    if not results:
        return

    panel = pygame.Rect(SIDEBAR_X, 570, 252, 232)
    draw_panel(surface, panel)
    draw_text(surface, font, "Comparison", panel.x + 18, panel.y + 16)
    draw_text(surface, small_font, "Lower fitness is better.", panel.x + 18, panel.y + 48, COLORS["muted"])

    header_y = panel.y + 82
    draw_text(surface, small_font, "Algorithm", panel.x + 18, header_y, COLORS["text"])
    draw_text(surface, small_font, "Fitness", panel.x + 124, header_y, COLORS["text"])
    draw_text(surface, small_font, "Time", panel.x + 194, header_y, COLORS["text"])

    for index, result in enumerate(results):
        y = header_y + 32 + index * 44
        row_color = (240, 253, 244) if index == 0 else (248, 250, 252)
        pygame.draw.rect(surface, row_color, (panel.x + 14, y - 8, panel.w - 28, 38), border_radius=7)
        draw_text(surface, small_font, result["name"], panel.x + 22, y, COLORS["text"])
        draw_text(surface, small_font, f"{result['fitness']:.2f}", panel.x + 124, y, COLORS["muted"])
        draw_text(surface, small_font, f"{result['elapsed']:.1f}s", panel.x + 194, y, COLORS["muted"])

    best = results[0]
    draw_text(surface, small_font, f"Best: {best['name']}", panel.x + 18, panel.y + 162, COLORS["green"])
    draw_text(surface, small_font, f"Timings: {best['timings']}", panel.x + 18, panel.y + 182, COLORS["muted"])


def draw_line_chart(surface, rect, values, title, color, small_font):
    pygame.draw.rect(surface, (249, 250, 251), rect, border_radius=10)
    pygame.draw.rect(surface, COLORS["border"], rect, 1, border_radius=10)
    draw_text(surface, small_font, title, rect.x + 12, rect.y + 8, COLORS["muted"])
    if len(values) < 2:
        draw_text(surface, small_font, "Collecting data...", rect.x + 12, rect.y + 30, COLORS["muted"])
        return

    chart = pygame.Rect(rect.x + 12, rect.y + 30, rect.w - 24, rect.h - 42)
    pygame.draw.line(surface, (229, 231, 235), (chart.x, chart.bottom), (chart.right, chart.bottom), 1)
    min_v = min(values)
    max_v = max(values)
    span = max(0.001, max_v - min_v)
    points = []
    recent = values[-48:]
    for index, value in enumerate(recent):
        x = chart.x + index * chart.w / max(1, len(recent) - 1)
        y = chart.bottom - ((value - min_v) / span) * chart.h
        points.append((int(x), int(y)))
    if len(points) > 1:
        pygame.draw.lines(surface, color, False, points, 3)


def draw_analytics_panel(surface, font, small_font, history, comparison_results, baseline_fitness, optimized_fitness, best_algorithm):
    panel = pygame.Rect(SIDEBAR_X, 558, SIDEBAR_W, 160)
    draw_panel(surface, panel)
    draw_text(surface, font, "Analytics", panel.x + 18, panel.y + 14)
    draw_line_chart(surface, pygame.Rect(panel.x + 16, panel.y + 44, panel.w - 32, 46), history["fitness"], "Fitness over time", COLORS["blue"], small_font)
    draw_line_chart(surface, pygame.Rect(panel.x + 16, panel.y + 96, panel.w - 32, 46), history["wait"], "Average wait time", COLORS["green"], small_font)

    if comparison_results:
        best = comparison_results[0]
        badge = pygame.Rect(panel.x + panel.w - 98, panel.y + 16, 72, 24)
        pygame.draw.rect(surface, (220, 252, 231), badge, border_radius=12)
        draw_text(surface, small_font, f"BEST {best['name']}", badge.centerx, badge.centery, COLORS["green"], center=True)
        y = panel.y + 142
        for result in comparison_results[:3]:
            row = pygame.Rect(panel.x + 16, y, panel.w - 32, 18)
            fill = (240, 253, 244) if result is best else (249, 250, 251)
            pygame.draw.rect(surface, fill, row, border_radius=7)
            draw_text(surface, small_font, result["name"], row.x + 8, row.y + 1, COLORS["text"])
            draw_text(surface, small_font, f"{result['fitness']:.2f}", row.x + 132, row.y + 1, COLORS["muted"])
            draw_text(surface, small_font, f"{result['elapsed']:.1f}s", row.x + 226, row.y + 1, COLORS["muted"])
            y += 20
    else:
        improvement = ((baseline_fitness - optimized_fitness) / baseline_fitness * 100) if baseline_fitness else 0
        draw_text(surface, small_font, f"{best_algorithm}: {improvement:.1f}% improvement", panel.x + 18, panel.y + 142, COLORS["muted"])


def run_optimizer(name):
    start = time.time()
    if name == "PSO":
        solution, fitness = run_pso()
    elif name == "GA":
        solution, fitness = run_ga(mutation_type=1, crossover_type=1, seed=42)
    else:
        solution, fitness = run_hybrid(mutation_type=1, crossover_type=1, selection_type=1, seed=42)
    return [float(value) for value in solution], float(fitness), time.time() - start


def calculate_project_fitness(timings):
    simulation = TrafficSimulation(num_intersections=6)
    individual = FitnessIndividual(timings)
    return float(calculate_fitness(individual, simulation))


def main():
    if pygame is None:
        print("Pygame is not installed. Install it with: pip install pygame-ce")
        return

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Traffic Signal Timing Simulator")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("Inter, Poppins, Segoe UI", 22, bold=True)
    small_font = pygame.font.SysFont("Inter, Poppins, Segoe UI", 16)
    car_images = load_car_images()
    environment = load_environment_assets()

    game = TrafficGame(BASELINE_TIMINGS, seed=7, density=0.08)
    game.set_images(car_images)

    optimized_solution = BASELINE_TIMINGS[:]
    baseline_fitness = calculate_project_fitness(BASELINE_TIMINGS)
    optimized_fitness = baseline_fitness
    message = "Choose a traffic level, then start the simulation or optimize the signal timings."
    simulation_active = False
    comparison_results = []
    selected_algorithm = "Hybrid"
    selected_scenario = "Medium"
    best_algorithm = "Baseline"
    history = {"fitness": [], "wait": []}
    frame_count = 0

    buttons = {
        "Start": pygame.Rect(SIDEBAR_X + 18, 414, 140, 42),
        "Stop": pygame.Rect(SIDEBAR_X + 170, 414, 140, 42),
        "Optimize": pygame.Rect(SIDEBAR_X + 18, 474, 292, 42),
    }
    algorithm_rects = {
        "PSO": pygame.Rect(SIDEBAR_X + 18, 216, 132, 34),
        "GA": pygame.Rect(SIDEBAR_X + 18, 258, 132, 34),
        "Hybrid": pygame.Rect(SIDEBAR_X + 18, 300, 132, 34),
    }
    scenario_rects = {
        "Low": pygame.Rect(SIDEBAR_X + 172, 216, 138, 34),
        "Medium": pygame.Rect(SIDEBAR_X + 172, 258, 138, 34),
        "High": pygame.Rect(SIDEBAR_X + 172, 300, 138, 34),
    }
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = pygame.mouse.get_pos()
                for name, rect in algorithm_rects.items():
                    if rect.collidepoint(pos):
                        selected_algorithm = name
                        message = f"{name} selected. Press Optimize to update the timing plan."

                for name, rect in scenario_rects.items():
                    if rect.collidepoint(pos):
                        selected_scenario = name
                        game.set_density(SCENARIOS[name])
                        game.reset(game.timings, game.mode_name)
                        history = {"fitness": [], "wait": []}
                        simulation_active = False
                        message = f"{name} traffic selected."

                for label, rect in buttons.items():
                    if not rect.collidepoint(pos):
                        continue
                    if label == "Start":
                        simulation_active = True
                        message = "Simulation running. Cars are moving through the six-intersection network."
                    elif label == "Stop":
                        simulation_active = False
                        message = "Simulation paused."
                    else:
                        algorithm = selected_algorithm
                        message = f"Running {algorithm} optimizer..."
                        pygame.display.set_caption(message)
                        comparison_results = compare_algorithms()
                        selected_result = next(result for result in comparison_results if result["name"] == algorithm)
                        optimized_solution = [float(value) for value in selected_result["timings"]]
                        optimized_fitness = selected_result["fitness"]
                        best_algorithm = algorithm
                        game.reset(optimized_solution, f"{algorithm} optimized")
                        history = {"fitness": [], "wait": []}
                        simulation_active = True
                        message = f"{algorithm} finished in {selected_result['elapsed']:.2f}s. Optimized signal timings are active."
                        pygame.display.set_caption("Traffic Signal Timing Simulator")

        if simulation_active:
            game.update()
            frame_count += 1
            if frame_count % 8 == 0:
                live = game.metrics()
                history["fitness"].append(live["fitness"])
                history["wait"].append(live["avg_wait"])
                history["fitness"] = history["fitness"][-72:]
                history["wait"] = history["wait"][-72:]

        draw_roads(screen, environment)
        draw_signals(screen, game, small_font)
        draw_cars(screen, game)
        metrics, improvement = draw_hud(screen, game, font, small_font, baseline_fitness, optimized_fitness, optimized_solution, message)
        draw_algorithm_menu(screen, font, small_font, selected_algorithm, algorithm_rects, selected_scenario, scenario_rects)
        mouse_pos = pygame.mouse.get_pos()
        draw_buttons(screen, font, buttons, mouse_pos)
        draw_status_message(screen, small_font, message)
        draw_analytics_panel(screen, font, small_font, history, comparison_results, baseline_fitness, optimized_fitness, best_algorithm)
        draw_bottom_stats(screen, small_font, metrics, improvement)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
