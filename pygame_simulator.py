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

# ============ SCREEN SETTINGS ============
WIDTH = 1600
HEIGHT = 920
FPS = 60

# ============ ROAD & CAR SETTINGS ============
INTERSECTIONS = 3
ROAD_W = 130
CAR_W = 34
CAR_H = 58
LANE_OFFSET = 28
CYCLE_LENGTH = 60
YELLOW_TIME = 4
SIGNAL_SPEED = 0.18

# ============ MAP LAYOUT ============
MAP_LEFT = 30
MAP_TOP = 50
MAP_WIDTH = 880
MAP_HEIGHT = 820

SIDEBAR_X = MAP_LEFT + MAP_WIDTH + 15
SIDEBAR_W = WIDTH - SIDEBAR_X - 15

INTERSECTION_POSITIONS = [
    (MAP_LEFT + 140, MAP_TOP + 400),
    (MAP_LEFT + 440, MAP_TOP + 400),
    (MAP_LEFT + 740, MAP_TOP + 400),
]

# 3 intersections x 2 phases:
# [I1_NS, I1_EW, I2_NS, I2_EW, I3_NS, I3_EW]
BASELINE_TIMINGS = [30, 30, 30, 30, 30, 30]
ASSET_DIR = Path(__file__).parent / "assets"
SCENARIOS = {
    "Low": 0.035,
    "Medium": 0.075,
    "High": 0.12,
}

# ============ LIGHT & CLEAN COLOR PALETTE ============
COLORS = {
    # Backgrounds - Light and clean
    "bg_main": (248, 250, 252),
    "bg_card": (255, 255, 255),
    "bg_card_light": (250, 251, 253),
    "bg_panel": (241, 245, 249),
    "bg_input": (248, 250, 252),
    "bg_sidebar": (255, 255, 255),
    
    # Accent colors - Modern tech
    "primary": (59, 130, 246),
    "primary_dark": (37, 99, 235),
    "primary_light": (96, 165, 250),
    
    "success": (34, 197, 94),
    "warning": (245, 158, 11),
    "danger": (239, 68, 68),
    "info": (6, 182, 212),
    
    # Text colors
    "text_main": (15, 23, 42),
    "text_secondary": (71, 85, 105),
    "text_muted": (148, 163, 184),
    "text_accent": (59, 130, 246),
    
    # Road colors - Realistic
    "road": (226, 232, 240),
    "road_border": (203, 213, 225),
    "road_line": (100, 116, 139),
    "road_dashed": (148, 163, 184),
    "sidewalk": (241, 245, 249),
    "asphalt": (203, 213, 225),
    "building": (255, 255, 255),
    "building_dark": (241, 245, 249),
    "building_light": (255, 255, 255),
    
    # Traffic lights
    "light_red": (239, 68, 68),
    "light_yellow": (245, 158, 11),
    "light_green": (34, 197, 94),
    "light_off": (203, 213, 225),
    
    # Borders and shadows
    "border": (226, 232, 240),
    "shadow": (0, 0, 0, 0.08),
}

class FitnessIndividual:
    def __init__(self, genome):
        self.genome = genome
        self.fitness = None

class Car:
    def __init__(self, road_id, direction, x, y, image, road_type):
        self.road_id = road_id
        self.direction = direction
        self.x = x
        self.y = y
        self.speed = random.uniform(1.8, 3.0)
        self.wait_time = 0
        self.stops = 0
        self.was_stopped = False
        self.done = False
        self.image = image
        self.road_type = road_type
        self.angle = 0
        self.update_angle()

    def update_angle(self):
        if self.road_type == 'horizontal':
            if self.direction == 1:
                self.angle = 270
            else:
                self.angle = 90
        else:
            if self.direction == 1:
                self.angle = 180
            else:
                self.angle = 0

    def get_rotated_image(self):
        return pygame.transform.rotate(self.image, self.angle)

    def collision_rect_at(self, x, y):
        if self.road_type == 'horizontal':
            width, height = CAR_H, CAR_W
        else:
            width, height = CAR_W, CAR_H
        center = (int(x + CAR_W // 2), int(y + CAR_H // 2))
        rect = pygame.Rect(0, 0, width, height)
        rect.center = center
        return rect

    def collision_rect(self):
        return self.collision_rect_at(self.x, self.y)

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
        self.mode_name = "Standard Pattern"
        self.roads = self.create_road_network()

    def create_road_network(self):
        roads = []
        y = INTERSECTION_POSITIONS[0][1]
        roads.append({
            'id': 0,
            'type': 'horizontal',
            'start': (INTERSECTION_POSITIONS[0][0], y),
            'end': (INTERSECTION_POSITIONS[-1][0], y),
            'intersections': tuple(range(INTERSECTIONS)),
        })

        for index, (x, y) in enumerate(INTERSECTION_POSITIONS, start=1):
            roads.append({
                'id': index,
                'type': 'vertical',
                'start': (x, MAP_TOP + 20),
                'end': (x, MAP_TOP + MAP_HEIGHT - 20),
                'intersections': (index - 1,),
            })
        return roads

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

    def light_state(self, intersection_id, road_type="horizontal"):
        phase_index = intersection_id * 2
        ns_time = max(10, float(self.timings[phase_index]))
        ew_time = max(10, float(self.timings[phase_index + 1]))
        cycle_length = ns_time + ew_time + YELLOW_TIME * 2
        signal_time = (self.step * SIGNAL_SPEED) % cycle_length

        horizontal_active = road_type == "horizontal"

        if signal_time < ns_time:
            return "red" if horizontal_active else "green"
        if signal_time < ns_time + YELLOW_TIME:
            return "red" if horizontal_active else "yellow"
        if signal_time < ns_time + YELLOW_TIME + ew_time:
            return "green" if horizontal_active else "red"
        return "yellow" if horizontal_active else "red"

    def car_can_pass(self, car, intersection_id, road_type):
        state = self.light_state(intersection_id, road_type)
        
        if state == "green":
            return True
        elif state == "yellow":
            return self.rng.random() < 0.3
        else:
            return False

    def lane_position(self, road, direction):
        if road['type'] == 'horizontal':
            _, y_center = road['start']
            # Left-to-right traffic uses the lower lane; right-to-left uses the upper lane.
            return y_center + LANE_OFFSET if direction == 1 else y_center - LANE_OFFSET

        x_center, _ = road['start']
        # Top-to-bottom traffic uses the left lane; bottom-to-top uses the right lane.
        return x_center - LANE_OFFSET if direction == 1 else x_center + LANE_OFFSET

    def front_sensor_rect(self, car, distance=86):
        rect = car.collision_rect()
        if car.road_type == 'horizontal':
            if car.direction == 1:
                return pygame.Rect(rect.right, rect.y - 8, distance, rect.h + 16)
            return pygame.Rect(rect.x - distance, rect.y - 8, distance, rect.h + 16)

        if car.direction == 1:
            return pygame.Rect(rect.x - 8, rect.bottom, rect.w + 16, distance)
        return pygame.Rect(rect.x - 8, rect.y - distance, rect.w + 16, distance)

    def has_vehicle_ahead(self, car):
        sensor = self.front_sensor_rect(car)
        for other in self.cars:
            if other is car or other.done:
                continue
            if sensor.colliderect(other.collision_rect()):
                return True
        return False

    def is_past_stop_line(self, car, stop_line):
        rect = car.collision_rect()
        if car.road_type == 'horizontal':
            return rect.left > stop_line if car.direction == 1 else rect.right < stop_line
        return rect.top > stop_line if car.direction == 1 else rect.bottom < stop_line

    def distance_to_stop_line(self, car, stop_line):
        rect = car.collision_rect()
        if car.road_type == 'horizontal':
            return stop_line - rect.right if car.direction == 1 else rect.left - stop_line
        return stop_line - rect.bottom if car.direction == 1 else rect.top - stop_line

    def stop_line_for(self, car, intersection_id):
        x, y = INTERSECTION_POSITIONS[intersection_id]
        margin = ROAD_W // 2 + 4
        if car.road_type == 'horizontal':
            return x - margin if car.direction == 1 else x + margin
        return y - margin if car.direction == 1 else y + margin

    def should_stop_for_signal(self, car, road):
        for inter_id in road['intersections']:
            stop_line = self.stop_line_for(car, inter_id)
            if self.is_past_stop_line(car, stop_line):
                continue
            distance = self.distance_to_stop_line(car, stop_line)
            if 0 < distance < 72 and not self.car_can_pass(car, inter_id, road['type']):
                return True
        return False

    def same_movement_lane(self, car, other):
        if other.done or other.road_type != car.road_type or other.direction != car.direction:
            return False
        if car.road_type == 'horizontal':
            return abs(other.y - car.y) < 8
        return abs(other.x - car.x) < 8

    def spawn_space_is_clear(self, road, direction, x, y):
        probe = pygame.Rect(0, 0, CAR_W, CAR_H)
        probe.center = (int(x + CAR_W // 2), int(y + CAR_H // 2))
        probe = probe.inflate(90, 90)
        for other in self.cars:
            if other.road_type != road['type'] or other.direction != direction:
                continue
            if road['type'] == 'horizontal':
                lane_center = self.lane_position(road, direction)
                if abs((other.y + CAR_H // 2) - lane_center) >= 8:
                    continue
            else:
                lane_center = self.lane_position(road, direction)
                if abs((other.x + CAR_W // 2) - lane_center) >= 8:
                    continue
            if probe.colliderect(other.collision_rect()):
                return False
        return True

    def spawn_car(self):
        if not self.roads:
            return
        road = self.rng.choice(self.roads)
        direction = self.rng.choice([-1, 1])
        
        if road['type'] == 'horizontal':
            if direction == 1:
                x = road['start'][0] - 120
            else:
                x = road['end'][0] + 80
            y = self.lane_position(road, direction) - CAR_H//2
        else:
            if direction == 1:
                y = road['start'][1] - 120
            else:
                y = road['end'][1] + 80
            x = self.lane_position(road, direction) - CAR_W//2

        if not self.spawn_space_is_clear(road, direction, x, y):
            return

        if self.car_images:
            key = self.rng.choice(list(self.car_images.keys()))
            image = self.car_images[key]
            self.cars.append(Car(road['id'], direction, x, y, image, road['type']))

    def update(self):
        self.step += 1
        self.spawn_meter += self.density
        while self.spawn_meter >= 1:
            self.spawn_meter -= 1
            self.spawn_car()

        def front_to_back_key(car):
            road = self.roads[car.road_id]
            position = car.x if road['type'] == 'horizontal' else car.y
            return (car.road_id, car.direction, -car.direction * position)

        for car in sorted(self.cars[:], key=front_to_back_key):
            if car.done:
                continue
            
            road = self.roads[car.road_id]
            moving = True

            if self.has_vehicle_ahead(car) or self.should_stop_for_signal(car, road):
                moving = False
            
            if moving and road['type'] == 'horizontal':
                next_x = car.x + car.direction * car.speed
                next_y = self.lane_position(road, car.direction) - CAR_H//2
                car.x = next_x
                car.y = next_y
            elif moving:
                next_y = car.y + car.direction * car.speed
                next_x = self.lane_position(road, car.direction) - CAR_W//2
                car.y = next_y
                car.x = next_x
            
            if moving:
                car.was_stopped = False
                if road['type'] == 'horizontal':
                    if car.x > road['end'][0] + 150 or car.x < road['start'][0] - 150:
                        car.done = True
                        self.completed += 1
                else:
                    if car.y > road['end'][1] + 150 or car.y < road['start'][1] - 150:
                        car.done = True
                        self.completed += 1
            else:
                car.wait_time += 1
                self.total_wait += 1
                if not car.was_stopped:
                    car.stops += 1
                    self.total_stops += 1
                car.was_stopped = True
            
            car.update_angle()
        
        self.cars = [c for c in self.cars if not c.done]

    def metrics(self):
        active = len(self.cars)
        avg_wait = self.total_wait / max(1, active + self.completed)
        waiting = sum(1 for car in self.cars if car.was_stopped)
        live_fitness = avg_wait + 1.4 * waiting + 0.35 * active + 0.15 * self.total_stops
        return {
            "active": active,
            "completed": self.completed,
            "avg_wait": avg_wait,
            "stops": self.total_stops,
            "fitness": live_fitness,
            "waiting": waiting,
            "throughput": self.completed,
            "efficiency": (self.completed / max(1, self.step)) * 100
        }

# ============ DRAWING FUNCTIONS ============
def draw_text(surface, font, text, x, y, color=COLORS["text_main"], center=False):
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

def draw_card(surface, rect, title=None):
    # Shadow
    shadow_rect = rect.inflate(4, 4)
    shadow_surf = pygame.Surface((shadow_rect.w, shadow_rect.h), pygame.SRCALPHA)
    pygame.draw.rect(shadow_surf, (0, 0, 0, 15), (2, 2, shadow_rect.w-2, shadow_rect.h-2), border_radius=12)
    surface.blit(shadow_surf, (shadow_rect.x-2, shadow_rect.y-2))
    
    # Card background
    pygame.draw.rect(surface, COLORS["bg_card"], rect, border_radius=12)
    pygame.draw.rect(surface, COLORS["border"], rect, 1, border_radius=12)
    
    if title:
        pygame.draw.rect(surface, COLORS["primary"], (rect.x, rect.y, rect.w, 3), border_radius=2)
        draw_text(surface, pygame.font.SysFont("Segoe UI", 16, bold=True), 
                 title, rect.x + 16, rect.y + 14, COLORS["text_main"])

def draw_modern_streets(screen, game):
    screen.fill(COLORS["bg_main"])

    map_container = pygame.Rect(MAP_LEFT - 10, MAP_TOP - 10, MAP_WIDTH + 20, MAP_HEIGHT + 20)
    shadow_surf = pygame.Surface((map_container.w, map_container.h), pygame.SRCALPHA)
    pygame.draw.rect(shadow_surf, (0, 0, 0, 18), (4, 4, map_container.w - 4, map_container.h - 4), border_radius=16)
    screen.blit(shadow_surf, (map_container.x - 2, map_container.y - 2))
    pygame.draw.rect(screen, COLORS["bg_card"], map_container, border_radius=16)
    pygame.draw.rect(screen, COLORS["border"], map_container, 1, border_radius=16)
    pygame.draw.rect(screen, (245, 158, 11), map_container.inflate(-4, -4), 4, border_radius=15)

    map_clip = map_container.inflate(-18, -18)
    old_clip = screen.get_clip()
    screen.set_clip(map_clip)

    # ====== LIGHT GREEN GRASS ======
    grass_color = (144, 238, 144)
    grass_dark = (124, 218, 124)
    for row in range(map_clip.h):
        ratio = row / max(1, map_clip.h - 1)
        color = tuple(int(grass_color[i] * (1 - ratio) + grass_dark[i] * ratio) for i in range(3))
        pygame.draw.line(screen, color, (map_clip.x, map_clip.y + row), (map_clip.right, map_clip.y + row))

    # Subtle grass texture dots
    for x in range(map_clip.x + 16, map_clip.right, 28):
        for y in range(map_clip.y + 16, map_clip.bottom, 28):
            pygame.draw.circle(screen, (104, 198, 104), (x, y), 2)

    road_color = (60, 60, 80)
    road_edge = (217, 119, 6)
    lane_color = (255, 215, 0)
    median_color = (255, 215, 0)

    road_rects = []
    horizontal_ys = sorted({road['start'][1] for road in game.roads if road['type'] == 'horizontal'})
    vertical_xs = sorted({road['start'][0] for road in game.roads if road['type'] == 'vertical'})
    for y in horizontal_ys:
        road_rects.append(pygame.Rect(map_clip.x - 8, y - ROAD_W // 2, map_clip.w + 16, ROAD_W))
    for x in vertical_xs:
        road_rects.append(pygame.Rect(x - ROAD_W // 2, map_clip.y - 8, ROAD_W, map_clip.h + 16))

    def is_clear_of_roads(rect, padding=18):
        return not any(rect.colliderect(road_rect.inflate(padding, padding)) for road_rect in road_rects)

    # ====== COLORFUL CITY BLOCKS ======
    blocks = [
        (230, 46, 122, 58, (255, 105, 180), (255, 255, 255)),
        (532, 46, 122, 58, (147, 112, 219), (255, 255, 255)),
        (22, 342, 46, 100, (60, 179, 113), (255, 255, 255)),
        (230, 342, 120, 104, (96, 165, 250), (255, 255, 255)),
        (532, 342, 120, 104, (251, 146, 60), (255, 255, 255)),
        (814, 342, 46, 100, (255, 99, 71), (255, 255, 255)),
        (230, 754, 122, 72, (255, 140, 0), (255, 255, 255)),
        (532, 754, 122, 72, (186, 85, 211), (255, 255, 255)),
    ]

    for index, (x, y, w, h, color, window_color) in enumerate(blocks):
        rect = pygame.Rect(MAP_LEFT + x, MAP_TOP + y, w, h)
        if not is_clear_of_roads(rect, padding=10):
            continue
        pygame.draw.rect(screen, (30, 30, 50), rect.inflate(8, 8), border_radius=12)
        pygame.draw.rect(screen, color, rect, border_radius=10)

        columns = 3 if rect.w >= 90 else 2
        rows = 3
        win_w = 18 if columns == 3 else 12
        win_h = 14
        win_gap_x = 8
        win_gap_y = 8
        start_x = rect.x + (rect.w - (columns * win_w + (columns - 1) * win_gap_x)) // 2
        start_y = rect.y + 16

        for row in range(rows):
            for col in range(columns):
                wx = start_x + col * (win_w + win_gap_x)
                wy = start_y + row * (win_h + win_gap_y)
                if wy + win_h > rect.bottom - 8:
                    continue
                win_rect = pygame.Rect(wx, wy, win_w, win_h)
                pygame.draw.rect(screen, window_color, win_rect, border_radius=3)
                pygame.draw.rect(screen, (230, 230, 230), (wx + 2, wy + 2, win_w - 4, 4), border_radius=2)

    def draw_lane_dashes_horizontal(y, x1, x2):
        for x in range(int(x1) + 30, int(x2) - 30, 50):
            pygame.draw.line(screen, lane_color, (x, y - 22), (x + 24, y - 22), 3)
            pygame.draw.line(screen, lane_color, (x, y + 22), (x + 24, y + 22), 3)

    def draw_lane_dashes_vertical(x, y1, y2):
        for y in range(int(y1) + 30, int(y2) - 30, 50):
            pygame.draw.line(screen, lane_color, (x - 22, y), (x - 22, y + 24), 3)
            pygame.draw.line(screen, lane_color, (x + 22, y), (x + 22, y + 24), 3)

    # Draw roads as continuous full-map corridors.
    for y in horizontal_ys:
        rect = pygame.Rect(map_clip.x - 8, y - ROAD_W // 2, map_clip.w + 16, ROAD_W)
        pygame.draw.rect(screen, road_edge, rect.inflate(8, 8), border_radius=0)
        pygame.draw.rect(screen, road_color, rect, border_radius=0)
        pygame.draw.line(screen, median_color, (rect.x + 12, y), (rect.right - 12, y), 4)
        draw_lane_dashes_horizontal(y, rect.x, rect.right)

    for x in vertical_xs:
        rect = pygame.Rect(x - ROAD_W // 2, map_clip.y - 8, ROAD_W, map_clip.h + 16)
        pygame.draw.rect(screen, road_edge, rect.inflate(8, 8), border_radius=0)
        pygame.draw.rect(screen, road_color, rect, border_radius=0)
        pygame.draw.line(screen, median_color, (x, rect.y + 12), (x, rect.bottom - 12), 4)
        draw_lane_dashes_vertical(x, rect.y, rect.bottom)

    # ====== TREES ON GRASS SIDES ONLY ======
    tree_candidates = [
        (58, 190), (250, 152), (520, 150), (842, 150),
        (58, 340), (300, 338), (620, 338), (855, 340),
        (58, 466), (300, 468), (620, 468), (855, 466),
        (58, 572), (300, 572), (620, 572), (855, 572),
        (250, 744), (520, 744), (842, 744),
        (238, 818), (520, 818), (842, 818),
        (24, 735), (866, 735), (24, 116), (866, 116),
    ]

    for local_x, local_y in tree_candidates:
        x = MAP_LEFT + local_x
        y = MAP_TOP + local_y
        tree_bounds = pygame.Rect(x - 18, y - 22, 36, 54)
        if not map_clip.contains(tree_bounds) or not is_clear_of_roads(tree_bounds, padding=8):
            continue
        pygame.draw.ellipse(screen, (0, 0, 0, 35), (x - 12, y + 10, 24, 12))
        pygame.draw.rect(screen, (101, 67, 33), (x - 4, y + 6, 8, 22), border_radius=2)
        pygame.draw.circle(screen, (22, 163, 74), (x, y - 4), 14)
        pygame.draw.circle(screen, (50, 205, 50), (x + 5, y - 10), 8)
    # ====== INTERSECTIONS ======
    intersection_size = ROAD_W + 10
    for i, (x, y) in enumerate(INTERSECTION_POSITIONS):
        junction = pygame.Rect(x - intersection_size // 2, y - intersection_size // 2, intersection_size, intersection_size)

        pygame.draw.rect(screen, (70, 70, 90), junction, border_radius=8)
        pygame.draw.rect(screen, (50, 50, 70), junction, 2, border_radius=8)

        # Crosswalks
        stripe_w = 6
        stripe_gap = 4
        crosswalk_len = 52

        for cx in range(x - crosswalk_len//2, x + crosswalk_len//2, stripe_w + stripe_gap):
            pygame.draw.rect(screen, (248, 250, 252), (cx, y - 56, stripe_w, 30), border_radius=1)
            pygame.draw.rect(screen, (248, 250, 252), (cx, y + 26, stripe_w, 30), border_radius=1)

        for cy in range(y - crosswalk_len//2, y + crosswalk_len//2, stripe_w + stripe_gap):
            pygame.draw.rect(screen, (248, 250, 252), (x - 56, cy, 30, stripe_w), border_radius=1)
            pygame.draw.rect(screen, (248, 250, 252), (x + 26, cy, 30, stripe_w), border_radius=1)

        # Label badge
        label = pygame.Rect(x - 18, y - 18, 36, 36)
        pygame.draw.ellipse(screen, (147, 112, 219), label)
        pygame.draw.ellipse(screen, (255, 255, 255), label, 2)
        draw_text(screen, pygame.font.SysFont("Segoe UI", 14, bold=True), f"#{i + 1}", x, y, (255, 255, 255), center=True)

    screen.set_clip(old_clip)


def draw_modern_signals(screen, game, small_font):
    def draw_signal_box(cx, cy, state, label):
        housing = pygame.Rect(cx - 14, cy - 24, 28, 48)
        pygame.draw.rect(screen, (255, 255, 255), housing, border_radius=8)
        pygame.draw.rect(screen, COLORS["border"], housing, 1, border_radius=8)
        lights = [
            ("red", COLORS["light_red"], cy - 13),
            ("yellow", COLORS["light_yellow"], cy),
            ("green", COLORS["light_green"], cy + 13),
        ]
        for name, color, ly in lights:
            active = state == name
            if active:
                glow = pygame.Surface((24, 24), pygame.SRCALPHA)
                pygame.draw.circle(glow, (*color, 100), (12, 12), 12)
                screen.blit(glow, (cx - 12, ly - 12))
            pygame.draw.circle(screen, color if active else COLORS["light_off"], (cx, ly), 6)
        draw_text(screen, small_font, label, cx, housing.y - 12, COLORS["text_accent"], center=True)

    for i, (x, y) in enumerate(INTERSECTION_POSITIONS):
        ns_state = game.light_state(i, "vertical")
        ew_state = game.light_state(i, "horizontal")

        pygame.draw.rect(screen, COLORS["text_muted"], (x - 2, y - 88, 4, 64), border_radius=2)
        draw_signal_box(x, y - 90, ns_state, f"I{i + 1} NS")

        pygame.draw.rect(screen, COLORS["text_muted"], (x - 88, y - 2, 64, 4), border_radius=2)
        draw_signal_box(x - 90, y, ew_state, f"I{i + 1} EW")
def draw_cars(screen, game):
    for car in game.cars:
        # Shadow
        shadow_surf = pygame.Surface((CAR_W + 8, CAR_H + 8), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surf, (0, 0, 0, 40), (2, CAR_H - 10, CAR_W + 4, 12))
        screen.blit(shadow_surf, (int(car.x) - 4, int(car.y) + 5))
        
        # Rotated car
        rotated = car.get_rotated_image()
        rect = rotated.get_rect(center=(int(car.x) + CAR_W//2, int(car.y) + CAR_H//2))
        screen.blit(rotated, rect)

def draw_sidebar(screen, font, small_font, game, optimized_solution, improvement,
                 selected_algo, algo_rects, selected_scene, scene_rects,
                 comparison_results, mouse_pos, message, history):
    sidebar_bg = pygame.Rect(SIDEBAR_X - 5, 0, SIDEBAR_W + 10, HEIGHT)
    pygame.draw.rect(screen, COLORS["bg_sidebar"], sidebar_bg)
    pygame.draw.line(screen, COLORS["border"], (SIDEBAR_X, 0), (SIDEBAR_X, HEIGHT), 1)

    title_font = pygame.font.SysFont("Segoe UI", 24, bold=True)
    value_font = pygame.font.SysFont("Segoe UI", 24, bold=True)

    header = pygame.Rect(SIDEBAR_X + 15, 20, SIDEBAR_W - 30, 86)
    draw_card(screen, header)
    draw_text(screen, title_font, "Traffic Signal Dashboard", header.x + 18, header.y + 18, COLORS["text_main"])
    draw_text(screen, small_font, f"Mode: {game.mode_name}", header.x + 18, header.y + 54, COLORS["text_secondary"])

    badge_color = COLORS["success"] if improvement > 0 else COLORS["warning"]
    badge = pygame.Rect(header.right - 112, header.y + 24, 86, 36)
    pygame.draw.rect(screen, badge_color, badge, border_radius=18)
    draw_text(screen, font, f"{improvement:.0f}%", badge.centerx, badge.centery, COLORS["bg_card"], center=True)

    metrics = game.metrics()
    cards_data = [
        ("Active", metrics["active"], COLORS["info"]),
        ("Avg Wait", f"{metrics['avg_wait']:.1f}s", COLORS["warning"]),
        ("Stops", metrics["stops"], COLORS["danger"]),
        ("Completed", metrics["completed"], COLORS["success"]),
        ("Efficiency", f"{metrics['efficiency']:.0f}%", COLORS["primary"]),
        ("Fitness", f"{metrics['fitness']:.1f}", COLORS["text_accent"]),
    ]

    card_w = (SIDEBAR_W - 46) // 3
    card_h = 78
    for idx, (label, value, color) in enumerate(cards_data):
        row = idx // 3
        col = idx % 3
        rect = pygame.Rect(SIDEBAR_X + 15 + col * (card_w + 8), 122 + row * (card_h + 10), card_w, card_h)
        pygame.draw.rect(screen, COLORS["bg_card"], rect, border_radius=12)
        pygame.draw.rect(screen, COLORS["border"], rect, 1, border_radius=12)
        pygame.draw.rect(screen, color, (rect.x, rect.y, rect.w, 4), border_radius=2)
        draw_text(screen, small_font, label, rect.x + 14, rect.y + 16, COLORS["text_muted"])
        draw_text(screen, value_font, str(value), rect.x + 14, rect.y + 40, color)

    timing_rect = pygame.Rect(SIDEBAR_X + 15, 292, SIDEBAR_W - 30, 66)
    draw_card(screen, timing_rect)
    draw_text(screen, small_font, "Signal Timing Plan", timing_rect.x + 16, timing_rect.y + 12, COLORS["text_secondary"])
    first = "   ".join([f"I{i+1} NS:{int(optimized_solution[i * 2])}s" for i in range(INTERSECTIONS)])
    second = "   ".join([f"I{i+1} EW:{int(optimized_solution[i * 2 + 1])}s" for i in range(INTERSECTIONS)])
    draw_text(screen, small_font, first, timing_rect.x + 16, timing_rect.y + 34, COLORS["text_main"])
    draw_text(screen, small_font, second, timing_rect.x + 16, timing_rect.y + 50, COLORS["text_main"])

    control_rect = pygame.Rect(SIDEBAR_X + 15, 374, SIDEBAR_W - 30, 168)
    draw_card(screen, control_rect, "Controls")
    draw_text(screen, small_font, "Algorithm", control_rect.x + 18, control_rect.y + 46, COLORS["text_secondary"])
    draw_text(screen, small_font, "Traffic Level", control_rect.x + 328, control_rect.y + 46, COLORS["text_secondary"])

    for options, selected_value in ((algo_rects, selected_algo), (scene_rects, selected_scene)):
        for name, rect in options.items():
            hovered = rect.collidepoint(mouse_pos)
            selected = name == selected_value
            bg = COLORS["primary"] if selected else COLORS["bg_input"]
            if hovered and not selected:
                bg = COLORS["bg_panel"]
            pygame.draw.rect(screen, bg, rect, border_radius=9)
            pygame.draw.rect(screen, COLORS["primary"] if selected else COLORS["border"], rect, 1, border_radius=9)
            text_color = COLORS["bg_card"] if selected else COLORS["text_secondary"]
            draw_text(screen, small_font, name, rect.centerx, rect.centery, text_color, center=True)

    btn_y = 562
    btn_w = (SIDEBAR_W - 46) // 3
    btns = [("START", COLORS["success"]), ("STOP", COLORS["danger"]), ("OPTIMIZE", COLORS["primary"])]
    for idx, (label, color) in enumerate(btns):
        btn_rect = pygame.Rect(SIDEBAR_X + 15 + idx * (btn_w + 8), btn_y, btn_w, 44)
        hovered = btn_rect.collidepoint(mouse_pos)
        bg = color if not hovered else tuple(min(255, c + 24) for c in color)
        pygame.draw.rect(screen, (0, 0, 0, 18), btn_rect.move(0, 3), border_radius=13)
        pygame.draw.rect(screen, bg, btn_rect, border_radius=13)
        draw_text(screen, font, label, btn_rect.centerx, btn_rect.centery, COLORS["bg_card"], center=True)

    status_rect = pygame.Rect(SIDEBAR_X + 15, 624, SIDEBAR_W - 30, 34)
    pygame.draw.rect(screen, COLORS["bg_panel"], status_rect, border_radius=12)
    pygame.draw.circle(screen, COLORS["primary"], (status_rect.x + 18, status_rect.centery), 4)
    draw_text(screen, small_font, fit_text(small_font, message, status_rect.w - 48),
              status_rect.x + 32, status_rect.y + 9, COLORS["text_secondary"])

    analytics_rect = pygame.Rect(SIDEBAR_X + 15, 674, SIDEBAR_W - 30, 216)
    draw_card(screen, analytics_rect, "Analytics")

    def draw_chart(rect, values, title, color):
        pygame.draw.rect(screen, COLORS["bg_input"], rect, border_radius=10)
        pygame.draw.rect(screen, COLORS["border"], rect, 1, border_radius=10)
        draw_text(screen, small_font, title, rect.x + 12, rect.y + 8, COLORS["text_secondary"])
        if len(values) < 2:
            draw_text(screen, small_font, "Collecting data...", rect.x + 12, rect.y + 32, COLORS["text_muted"])
            return
        chart = pygame.Rect(rect.x + 12, rect.y + 34, rect.w - 24, rect.h - 44)
        recent = values[-40:]
        min_v = min(recent)
        max_v = max(recent)
        span = max(0.001, max_v - min_v)
        points = []
        for i, value in enumerate(recent):
            x = chart.x + i * chart.w / max(1, len(recent) - 1)
            y = chart.bottom - ((value - min_v) / span) * chart.h
            points.append((int(x), int(y)))
        pygame.draw.line(screen, COLORS["border"], (chart.x, chart.bottom), (chart.right, chart.bottom), 1)
        pygame.draw.lines(screen, color, False, points, 3)

    draw_chart(pygame.Rect(analytics_rect.x + 16, analytics_rect.y + 46, analytics_rect.w // 2 - 24, 74),
               history["fitness"], "Fitness over time", COLORS["primary"])
    draw_chart(pygame.Rect(analytics_rect.x + analytics_rect.w // 2 + 8, analytics_rect.y + 46, analytics_rect.w // 2 - 24, 74),
               history["wait"], "Average wait time", COLORS["success"])

    table_y = analytics_rect.y + 136
    if comparison_results:
        best = comparison_results[0]
        for i, res in enumerate(comparison_results[:3]):
            row = pygame.Rect(analytics_rect.x + 16, table_y + i * 22, analytics_rect.w - 32, 20)
            pygame.draw.rect(screen, (240, 253, 244) if res is best else COLORS["bg_input"], row, border_radius=7)
            prefix = "BEST " if res is best else ""
            draw_text(screen, small_font, f"{prefix}{res['name']}", row.x + 10, row.y + 3,
                      COLORS["success"] if res is best else COLORS["text_secondary"])
            draw_text(screen, small_font, f"Fitness {res['fitness']:.1f}", row.x + 220, row.y + 3, COLORS["text_main"])
            draw_text(screen, small_font, f"{res['elapsed']:.1f}s", row.right - 58, row.y + 3, COLORS["text_muted"])
    else:
        draw_text(screen, small_font, "Run Optimize to compare PSO, GA, and Hybrid.",
                  analytics_rect.x + 16, table_y + 8, COLORS["text_muted"])
def load_car_images():
    ASSET_DIR.mkdir(exist_ok=True)
    specs = [
        ("red", (239, 68, 68)),
        ("green", (34, 197, 94)),
        ("blue", (37, 99, 235)),
        ("yellow", (245, 158, 11)),
        ("dark", (31, 41, 55)),
    ]
    images = {}
    for name, color in specs:
        path = ASSET_DIR / f"dashboard_car_{name}.png"
        if not path.exists():
            create_car_asset(path, color)
        img = pygame.image.load(str(path))
        img = pygame.transform.smoothscale(img, (CAR_W, CAR_H))
        if pygame.display.get_surface() is not None:
            img = img.convert_alpha()
        images[name] = img
    return images

def create_car_asset(path, body_color):
    surf = pygame.Surface((CAR_W, CAR_H), pygame.SRCALPHA)
    shade = tuple(max(0, c - 60) for c in body_color)
    pygame.draw.ellipse(surf, (0, 0, 0, 50), (4, CAR_H - 9, CAR_W - 8, 9))
    pygame.draw.rect(surf, (255, 255, 255), (4, 4, CAR_W - 8, CAR_H - 8), border_radius=12)
    pygame.draw.rect(surf, shade, (7, 10, CAR_W - 14, CAR_H - 20), border_radius=9)
    pygame.draw.rect(surf, body_color, (7, 7, CAR_W - 14, CAR_H - 18), border_radius=10)
    pygame.draw.rect(surf, (219, 234, 254), (10, 12, CAR_W - 20, 12), border_radius=4)
    pygame.draw.rect(surf, (219, 234, 254), (10, 34, CAR_W - 20, 10), border_radius=4)
    pygame.draw.rect(surf, (17, 24, 39), (5, 14, 5, 11), border_radius=2)
    pygame.draw.rect(surf, (17, 24, 39), (CAR_W - 10, 14, 5, 11), border_radius=2)
    pygame.draw.rect(surf, (17, 24, 39), (5, 39, 5, 11), border_radius=2)
    pygame.draw.rect(surf, (17, 24, 39), (CAR_W - 10, 39, 5, 11), border_radius=2)
    pygame.draw.circle(surf, (254, 240, 138), (12, 8), 2)
    pygame.draw.circle(surf, (254, 240, 138), (CAR_W - 12, 8), 2)
    pygame.image.save(surf, path)

def run_optimizer(name):
    start = time.time()
    if name == "PSO":
        solution, fitness = run_pso()
    elif name == "GA":
        solution, fitness = run_ga(mutation_type=1, crossover_type=1, seed=42)
    else:
        solution, fitness = run_hybrid(mutation_type=1, crossover_type=1, selection_type=1, seed=42)
    return [float(v) for v in solution], float(fitness), time.time() - start

def calculate_project_fitness(timings):
    simulation = TrafficSimulation(num_intersections=6)
    individual = FitnessIndividual(timings)
    return float(calculate_fitness(individual, simulation))

def compare_algorithms():
    results = []
    for name in ["PSO", "GA", "Hybrid"]:
        solution, fitness, elapsed = run_optimizer(name)
        results.append({"name": name, "fitness": fitness, "elapsed": elapsed, "timings": [int(round(v)) for v in solution]})
    return sorted(results, key=lambda x: x["fitness"])

def main():
    global game
    
    if pygame is None:
        print("Pygame not installed. Install: pip install pygame-ce")
        return

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Smart Traffic Signal Optimization System")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("Segoe UI", 16, bold=True)
    small_font = pygame.font.SysFont("Segoe UI", 12)
    
    car_images = load_car_images()
    game = TrafficGame(BASELINE_TIMINGS, seed=7, density=0.075)
    game.set_images(car_images)

    optimized_solution = BASELINE_TIMINGS[:]
    baseline_fitness = calculate_project_fitness(BASELINE_TIMINGS)
    optimized_fitness = baseline_fitness
    message = "System ready | Select algorithm and traffic level"
    simulation_active = False
    comparison_results = []
    selected_algorithm = "Hybrid"
    selected_scenario = "Medium"
    history = {"fitness": [], "wait": []}
    frame_count = 0
    
    algo_rects = {
        "PSO": pygame.Rect(SIDEBAR_X + 33, 452, 260, 30),
        "GA": pygame.Rect(SIDEBAR_X + 33, 488, 260, 30),
        "Hybrid": pygame.Rect(SIDEBAR_X + 33, 524, 260, 30),
    }
    
    scene_rects = {
        "Low": pygame.Rect(SIDEBAR_X + 343, 452, 260, 30),
        "Medium": pygame.Rect(SIDEBAR_X + 343, 488, 260, 30),
        "High": pygame.Rect(SIDEBAR_X + 343, 524, 260, 30),
    }
    
    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                for name, rect in algo_rects.items():
                    if rect.collidepoint(mouse_pos):
                        selected_algorithm = name
                        existing_result = next((result for result in comparison_results if result["name"] == name), None)
                        if existing_result:
                            optimized_solution = [float(value) for value in existing_result["timings"]]
                            optimized_fitness = existing_result["fitness"]
                            game.reset(optimized_solution, name)
                            history = {"fitness": [], "wait": []}
                            frame_count = 0
                            simulation_active = True
                            message = f"{name} timings applied"
                        else:
                            message = f"{name} algorithm selected | Press Optimize to apply timings"
                for name, rect in scene_rects.items():
                    if rect.collidepoint(mouse_pos):
                        selected_scenario = name
                        game.set_density(SCENARIOS[name])
                        game.reset(game.timings, game.mode_name)
                        history = {"fitness": [], "wait": []}
                        frame_count = 0
                        message = f"{name} traffic volume selected"
                
                btn_y = 562
                btn_w = (SIDEBAR_W - 46) // 3
                for i in range(3):
                    btn_rect = pygame.Rect(SIDEBAR_X + 15 + i * (btn_w + 8), btn_y, btn_w, 44)
                    if btn_rect.collidepoint(mouse_pos):
                        if i == 0:
                            simulation_active = True
                            message = "Simulation running | Traffic flow active"
                        elif i == 1:
                            simulation_active = False
                            message = "Simulation paused"
                        else:
                            message = f"Running {selected_algorithm} optimization..."
                            comparison_results = compare_algorithms()
                            selected = next(r for r in comparison_results if r["name"] == selected_algorithm)
                            optimized_solution = [float(v) for v in selected["timings"]]
                            optimized_fitness = selected["fitness"]
                            game.reset(optimized_solution, f"{selected_algorithm}")
                            history = {"fitness": [], "wait": []}
                            frame_count = 0
                            simulation_active = True
                            message = f"{selected_algorithm} complete | New timings active"
        
        if simulation_active:
            game.update()
            frame_count += 1
            if frame_count % 8 == 0:
                live = game.metrics()
                history["fitness"].append(live["fitness"])
                history["wait"].append(live["avg_wait"])
                history["fitness"] = history["fitness"][-72:]
                history["wait"] = history["wait"][-72:]
        
        draw_modern_streets(screen, game)
        draw_modern_signals(screen, game, small_font)
        draw_cars(screen, game)
        
        improvement = ((baseline_fitness - optimized_fitness) / baseline_fitness * 100) if baseline_fitness else 0
        draw_sidebar(screen, font, small_font, game, optimized_solution, improvement,
                    selected_algorithm, algo_rects, selected_scenario, scene_rects,
                    comparison_results, mouse_pos, message, history)
        
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()

if __name__ == "__main__":
    main()
