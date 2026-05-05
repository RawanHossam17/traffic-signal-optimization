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
INTERSECTIONS = 6
ROAD_W = 130
CAR_W = 34
CAR_H = 58
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
    (MAP_LEFT + 100, MAP_TOP + 150),
    (MAP_LEFT + 380, MAP_TOP + 100),
    (MAP_LEFT + 700, MAP_TOP + 180),
    (MAP_LEFT + 150, MAP_TOP + 580),
    (MAP_LEFT + 450, MAP_TOP + 650),
    (MAP_LEFT + 680, MAP_TOP + 550),
]

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
                self.angle = 0
            else:
                self.angle = 180
        else:
            if self.direction == 1:
                self.angle = 90
            else:
                self.angle = 270

    def get_rotated_image(self):
        return pygame.transform.rotate(self.image, self.angle)

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
        connections = [
            (0, 1), (1, 2), (0, 3), (1, 4), (2, 5), (3, 4), (4, 5)
        ]
        for i, (a, b) in enumerate(connections):
            x1, y1 = INTERSECTION_POSITIONS[a]
            x2, y2 = INTERSECTION_POSITIONS[b]
            if abs(x1 - x2) > abs(y1 - y2):
                roads.append({
                    'id': i,
                    'type': 'horizontal',
                    'start': (min(x1, x2), (y1 + y2)//2),
                    'end': (max(x1, x2), (y1 + y2)//2),
                    'intersections': (a, b)
                })
            else:
                roads.append({
                    'id': i,
                    'type': 'vertical',
                    'start': ((x1 + x2)//2, min(y1, y2)),
                    'end': ((x1 + x2)//2, max(y1, y2)),
                    'intersections': (a, b)
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

    def light_state(self, intersection_id):
        signal_time = (self.step * SIGNAL_SPEED) % CYCLE_LENGTH
        green_time = self.timings[intersection_id]
        
        if signal_time < green_time:
            return "green"
        elif signal_time < green_time + YELLOW_TIME:
            return "yellow"
        else:
            return "red"

    def car_can_pass(self, car, intersection_id):
        state = self.light_state(intersection_id)
        
        if state == "green":
            return True
        elif state == "yellow":
            return self.rng.random() < 0.3
        else:
            return False

    def spawn_car(self):
        if not self.roads:
            return
        road = self.rng.choice(self.roads)
        direction = self.rng.choice([-1, 1])
        
        if road['type'] == 'horizontal':
            if direction == 1:
                x = road['start'][0] - 120
                y = road['start'][1] - CAR_H//2
            else:
                x = road['end'][0] + 80
                y = road['end'][1] - CAR_H//2
        else:
            if direction == 1:
                x = road['start'][0] - CAR_W//2
                y = road['start'][1] - 120
            else:
                x = road['end'][0] - CAR_W//2
                y = road['end'][1] + 80
                
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

        for car in self.cars[:]:
            if car.done:
                continue
            
            road = self.roads[car.road_id]
            moving = True
            
            if road['type'] == 'horizontal':
                car.x += car.direction * car.speed
                car.y = road['start'][1] - CAR_H//2
                
                for inter_id in road['intersections']:
                    ix, iy = INTERSECTION_POSITIONS[inter_id]
                    if abs(car.x + CAR_W//2 - ix) < 35:
                        if not self.car_can_pass(car, inter_id):
                            moving = False
                            car.x -= car.direction * car.speed
                            break
            else:
                car.y += car.direction * car.speed
                car.x = road['start'][0] - CAR_W//2
                
                for inter_id in road['intersections']:
                    ix, iy = INTERSECTION_POSITIONS[inter_id]
                    if abs(car.y + CAR_H//2 - iy) < 35:
                        if not self.car_can_pass(car, inter_id):
                            moving = False
                            car.y -= car.direction * car.speed
                            break
            
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
        live_fitness = avg_wait + 0.5 * self.total_stops + 0.2 * active
        return {
            "active": active,
            "completed": self.completed,
            "avg_wait": avg_wait,
            "stops": self.total_stops,
            "fitness": live_fitness,
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
    
    # Map container with shadow
    map_container = pygame.Rect(MAP_LEFT - 10, MAP_TOP - 10, MAP_WIDTH + 20, MAP_HEIGHT + 20)
    shadow_surf = pygame.Surface((map_container.w, map_container.h), pygame.SRCALPHA)
    pygame.draw.rect(shadow_surf, (0, 0, 0, 20), (4, 4, map_container.w-4, map_container.h-4), border_radius=16)
    screen.blit(shadow_surf, (map_container.x-2, map_container.y-2))
    pygame.draw.rect(screen, COLORS["bg_card"], map_container, border_radius=16)
    pygame.draw.rect(screen, COLORS["border"], map_container, 1, border_radius=16)
    
    # Buildings (modern rectangles)
    for i, (x, y) in enumerate(INTERSECTION_POSITIONS):
        corners = [(x-85, y-85), (x+35, y-85), (x-85, y+35), (x+35, y+35)]
        for bx, by in corners:
            building_w = random.randint(35, 45)
            building_h = random.randint(35, 45)
            building = pygame.Rect(bx, by, building_w, building_h)
            pygame.draw.rect(screen, COLORS["building"], building, border_radius=6)
            pygame.draw.rect(screen, COLORS["border"], building, 1, border_radius=6)
            # Windows
            for wx in range(2):
                for wy in range(2):
                    pygame.draw.rect(screen, COLORS["text_muted"], 
                                   (bx + 8 + wx*15, by + 10 + wy*15, 5, 5), border_radius=1)
    
    # Draw roads
    for road in game.roads:
        if road['type'] == 'horizontal':
            start_x, y_center = road['start']
            end_x, _ = road['end']
            road_rect = pygame.Rect(min(start_x, end_x) - 10, y_center - ROAD_W//2, 
                                   abs(end_x - start_x) + 20, ROAD_W)
            pygame.draw.rect(screen, COLORS["road"], road_rect)
            pygame.draw.rect(screen, COLORS["road_border"], road_rect, 2)
            y_line = y_center
            pygame.draw.line(screen, COLORS["road_line"], 
                           (min(start_x, end_x) + 20, y_line), 
                           (max(start_x, end_x) - 20, y_line), 3)
        else:
            x_center, start_y = road['start']
            _, end_y = road['end']
            road_rect = pygame.Rect(x_center - ROAD_W//2, min(start_y, end_y) - 10, 
                                   ROAD_W, abs(end_y - start_y) + 20)
            pygame.draw.rect(screen, COLORS["road"], road_rect)
            pygame.draw.rect(screen, COLORS["road_border"], road_rect, 2)
            x_line = x_center
            pygame.draw.line(screen, COLORS["road_line"], 
                           (x_line, min(start_y, end_y) + 20), 
                           (x_line, max(start_y, end_y) - 20), 3)
    
    # Intersections
    for i, (x, y) in enumerate(INTERSECTION_POSITIONS):
        # Intersection circle
        pygame.draw.circle(screen, COLORS["asphalt"], (x, y), 45)
        pygame.draw.circle(screen, COLORS["road_border"], (x, y), 45, 2)
        # Intersection label
        label_bg = pygame.Rect(x - 15, y - 12, 30, 24)
        pygame.draw.rect(screen, COLORS["primary"], label_bg, border_radius=12)
        draw_text(screen, pygame.font.SysFont("Segoe UI", 14, bold=True), 
                 str(i+1), x, y, COLORS["bg_card"], center=True)

def draw_modern_signals(screen, game, small_font):
    for i, (x, y) in enumerate(INTERSECTION_POSITIONS):
        state = game.light_state(i)
        
        # Modern traffic light pole
        pole_x = x
        pole_y = y - 65
        
        # Slim pole
        pygame.draw.rect(screen, COLORS["text_muted"], (pole_x - 2, pole_y, 4, 55), border_radius=2)
        
        # Modern housing
        housing = pygame.Rect(pole_x - 14, pole_y - 38, 28, 48)
        pygame.draw.rect(screen, COLORS["bg_card"], housing, border_radius=8)
        pygame.draw.rect(screen, COLORS["border"], housing, 1, border_radius=8)
        
        # Lights (horizontal arrangement for modern look)
        light_x_positions = [pole_x - 8, pole_x - 1, pole_x + 6]
        light_names = ["red", "yellow", "green"]
        light_colors = [COLORS["light_red"], COLORS["light_yellow"], COLORS["light_green"]]
        
        for idx, (name, color, lx) in enumerate(zip(light_names, light_colors, light_x_positions)):
            active = (state == name)
            ly = pole_y - 22
            if active:
                # Glow effect
                glow = pygame.Surface((20, 20), pygame.SRCALPHA)
                glow_color = (*color, 100)
                pygame.draw.circle(glow, glow_color, (10, 10), 10)
                screen.blit(glow, (lx - 8, ly - 8))
            pygame.draw.circle(screen, color if active else COLORS["light_off"], (lx + 4, ly), 6)
        
        # Signal number
        draw_text(screen, small_font, f"S{i+1}", pole_x - 20, pole_y - 15, COLORS["text_accent"])

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
                 comparison_results, mouse_pos, message):
    
    # Sidebar background
    sidebar_bg = pygame.Rect(SIDEBAR_X - 5, 0, SIDEBAR_W + 10, HEIGHT)
    pygame.draw.rect(screen, COLORS["bg_sidebar"], sidebar_bg)
    pygame.draw.line(screen, COLORS["border"], (SIDEBAR_X, 0), (SIDEBAR_X, HEIGHT), 1)
    
    # Header
    header = pygame.Rect(SIDEBAR_X + 10, 20, SIDEBAR_W - 20, 80)
    draw_card(screen, header)
    draw_text(screen, font, "🚦 Traffic Control", SIDEBAR_X + 30, 48, COLORS["primary"])
    draw_text(screen, small_font, game.mode_name, SIDEBAR_X + 30, 78, COLORS["text_secondary"])
    
    # Improvement badge
    badge_color = COLORS["success"] if improvement > 0 else COLORS["warning"]
    badge = pygame.Rect(SIDEBAR_X + SIDEBAR_W - 95, 35, 80, 35)
    pygame.draw.rect(screen, badge_color, badge, border_radius=18)
    draw_text(screen, small_font, f"{improvement:.0f}%", badge.centerx, badge.centery, COLORS["bg_card"], center=True)
    
    # Metrics grid
    metrics = game.metrics()
    cards_data = [
        ("🚗", "Active", metrics["active"], COLORS["info"]),
        ("⏱️", "Avg Wait", f"{metrics['avg_wait']:.1f}s", COLORS["warning"]),
        ("🛑", "Stops", metrics["stops"], COLORS["danger"]),
        ("✅", "Completed", metrics["completed"], COLORS["success"]),
        ("⚡", "Efficiency", f"{metrics['efficiency']:.0f}%", COLORS["primary"]),
        ("📊", "Fitness", f"{metrics['fitness']:.0f}", COLORS["text_accent"]),
    ]
    
    card_w = (SIDEBAR_W - 35) // 3
    card_h = 70
    
    for idx, (icon, label, val, color) in enumerate(cards_data):
        row = idx // 3
        col = idx % 3
        x = SIDEBAR_X + 15 + col * (card_w + 8)
        y = 115 + row * (card_h + 8)
        rect = pygame.Rect(x, y, card_w, card_h)
        
        # Card
        pygame.draw.rect(screen, COLORS["bg_card"], rect, border_radius=10)
        pygame.draw.rect(screen, COLORS["border"], rect, 1, border_radius=10)
        
        # Colored top line
        pygame.draw.rect(screen, color, (rect.x, rect.y, rect.w, 3), border_radius=2)
        
        draw_text(screen, pygame.font.SysFont("Segoe UI Emoji", 20), icon, rect.x + 12, rect.y + 12)
        val_font = pygame.font.SysFont("Segoe UI", 22, bold=True)
        draw_text(screen, val_font, str(val), rect.x + 12, rect.y + 38, color)
        draw_text(screen, small_font, label, rect.x + 12, rect.y + 58, COLORS["text_muted"])
    
    # Signal timings display
    timing_rect = pygame.Rect(SIDEBAR_X + 15, 320, SIDEBAR_W - 30, 45)
    pygame.draw.rect(screen, COLORS["bg_panel"], timing_rect, border_radius=10)
    pygame.draw.rect(screen, COLORS["border"], timing_rect, 1, border_radius=10)
    timings_str = " | ".join([f"S{i+1}:{int(v)}s" for i, v in enumerate(optimized_solution[:3])])
    timings_str2 = " | ".join([f"S{i+1}:{int(v)}s" for i, v in enumerate(optimized_solution[3:])])
    draw_text(screen, small_font, timings_str, timing_rect.centerx, timing_rect.y + 16, COLORS["text_main"], center=True)
    draw_text(screen, small_font, timings_str2, timing_rect.centerx, timing_rect.y + 32, COLORS["text_main"], center=True)
    
    # Algorithm selector
    algo_panel = pygame.Rect(SIDEBAR_X + 15, 380, SIDEBAR_W - 30, 95)
    draw_card(screen, algo_panel, "Algorithm")
    for name, rect in algo_rects.items():
        hovered = rect.collidepoint(mouse_pos)
        selected = name == selected_algo
        bg = COLORS["primary"] if selected else COLORS["bg_input"]
        if hovered and not selected:
            bg = COLORS["bg_panel"]
        pygame.draw.rect(screen, bg, rect, border_radius=8)
        pygame.draw.rect(screen, COLORS["primary"] if selected else COLORS["border"], rect, 1, border_radius=8)
        text_color = COLORS["bg_card"] if selected else COLORS["text_secondary"]
        draw_text(screen, small_font, name, rect.centerx, rect.centery, text_color, center=True)
    
    # Scenario selector
    scene_panel = pygame.Rect(SIDEBAR_X + 15, 490, SIDEBAR_W - 30, 95)
    draw_card(screen, scene_panel, "Traffic Volume")
    for name, rect in scene_rects.items():
        hovered = rect.collidepoint(mouse_pos)
        selected = name == selected_scene
        bg = COLORS["primary"] if selected else COLORS["bg_input"]
        if hovered and not selected:
            bg = COLORS["bg_panel"]
        pygame.draw.rect(screen, bg, rect, border_radius=8)
        pygame.draw.rect(screen, COLORS["primary"] if selected else COLORS["border"], rect, 1, border_radius=8)
        text_color = COLORS["bg_card"] if selected else COLORS["text_secondary"]
        draw_text(screen, small_font, name, rect.centerx, rect.centery, text_color, center=True)
    
    # Buttons
    btn_y = 600
    btns = [("▶ START", COLORS["success"]), ("⏹ STOP", COLORS["danger"]), ("⚡ OPTIMIZE", COLORS["primary"])]
    btn_w = (SIDEBAR_W - 45) // 3
    
    for idx, (label, color) in enumerate(btns):
        btn_rect = pygame.Rect(SIDEBAR_X + 15 + idx * (btn_w + 8), btn_y, btn_w, 40)
        hovered = btn_rect.collidepoint(mouse_pos)
        bg = color if not hovered else tuple(min(255, c + 30) for c in color)
        pygame.draw.rect(screen, bg, btn_rect, border_radius=10)
        draw_text(screen, font, label, btn_rect.centerx, btn_rect.centery, COLORS["bg_card"], center=True)
    
    # Status
    status_rect = pygame.Rect(SIDEBAR_X + 15, 655, SIDEBAR_W - 30, 35)
    pygame.draw.rect(screen, COLORS["bg_panel"], status_rect, border_radius=10)
    draw_text(screen, small_font, message[:50], status_rect.x + 12, status_rect.centery, COLORS["text_secondary"])
    
    # Comparison results
    if comparison_results:
        comp_rect = pygame.Rect(SIDEBAR_X + 15, 705, SIDEBAR_W - 30, 160)
        draw_card(screen, comp_rect, "Algorithm Comparison")
        y = 745
        for i, res in enumerate(comparison_results[:3]):
            row_rect = pygame.Rect(SIDEBAR_X + 20, y-2, SIDEBAR_W - 40, 38)
            row_bg = COLORS["bg_input"] if i % 2 == 0 else COLORS["bg_card_light"]
            pygame.draw.rect(screen, row_bg, row_rect, border_radius=8)
            
            if i == 0:
                draw_text(screen, small_font, "🏆", SIDEBAR_X + 30, y+8, COLORS["warning"])
                draw_text(screen, small_font, res["name"], SIDEBAR_X + 55, y+8, COLORS["success"])
            else:
                draw_text(screen, small_font, res["name"], SIDEBAR_X + 30, y+8, COLORS["text_secondary"])
            
            draw_text(screen, small_font, f"Fitness: {res['fitness']:.0f}", SIDEBAR_X + 160, y+8, COLORS["text_main"])
            draw_text(screen, small_font, f"Time: {res['elapsed']:.1f}s", SIDEBAR_X + 280, y+8, COLORS["text_secondary"])
            y += 42

def load_car_images():
    ASSET_DIR.mkdir(exist_ok=True)
    specs = [
        ("red", (220, 80, 80)),
        ("white", (255, 255, 255)),
        ("blue", (80, 120, 220)),
        ("silver", (200, 205, 210)),
        ("black", (60, 65, 75)),
    ]
    images = {}
    for name, color in specs:
        path = ASSET_DIR / f"car_{name}.png"
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
    # Body
    pygame.draw.rect(surf, shade, (4, 10, 26, 42), border_radius=8)
    pygame.draw.rect(surf, body_color, (3, 4, 28, 48), border_radius=10)
    # Windows
    pygame.draw.rect(surf, (100, 150, 200, 180), (7, 8, 20, 12), border_radius=3)
    # Wheels
    pygame.draw.circle(surf, (30, 35, 45), (8, 50), 5)
    pygame.draw.circle(surf, (30, 35, 45), (26, 50), 5)
    pygame.draw.circle(surf, (30, 35, 45), (8, 8), 5)
    pygame.draw.circle(surf, (30, 35, 45), (26, 8), 5)
    # Lights
    pygame.draw.circle(surf, (255, 200, 50), (28, 12), 2)
    pygame.draw.circle(surf, (255, 100, 100), (28, 46), 2)
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
    
    algo_rects = {
        "PSO": pygame.Rect(SIDEBAR_X + 30, 420, 85, 32),
        "GA": pygame.Rect(SIDEBAR_X + 125, 420, 85, 32),
        "Hybrid": pygame.Rect(SIDEBAR_X + 220, 420, 85, 32),
    }
    
    scene_rects = {
        "Low": pygame.Rect(SIDEBAR_X + 30, 530, 85, 32),
        "Medium": pygame.Rect(SIDEBAR_X + 125, 530, 85, 32),
        "High": pygame.Rect(SIDEBAR_X + 220, 530, 85, 32),
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
                        message = f"{name} algorithm selected"
                for name, rect in scene_rects.items():
                    if rect.collidepoint(mouse_pos):
                        selected_scenario = name
                        game.set_density(SCENARIOS[name])
                        game.reset(game.timings, game.mode_name)
                        message = f"{name} traffic volume selected"
                
                btn_y = 600
                btn_w = (SIDEBAR_W - 45) // 3
                for i in range(3):
                    btn_rect = pygame.Rect(SIDEBAR_X + 15 + i * (btn_w + 8), btn_y, btn_w, 40)
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
                            simulation_active = True
                            message = f"{selected_algorithm} complete | New timings active"
        
        if simulation_active:
            game.update()
        
        draw_modern_streets(screen, game)
        draw_modern_signals(screen, game, small_font)
        draw_cars(screen, game)
        
        improvement = ((baseline_fitness - optimized_fitness) / baseline_fitness * 100) if baseline_fitness else 0
        draw_sidebar(screen, font, small_font, game, optimized_solution, improvement,
                    selected_algorithm, algo_rects, selected_scenario, scene_rects,
                    comparison_results, mouse_pos, message)
        
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()

if __name__ == "__main__":
    main()