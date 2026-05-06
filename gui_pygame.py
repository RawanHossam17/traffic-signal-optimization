"""
Traffic Signal Optimisation – Pygame GUI
AI420 Evolutionary Algorithms – Spring 2026

Standalone GUI: copies simulation logic inline so it runs without the full
package structure.  Drop this file next to pso.py / traffic_simulation.py
and run:
    python gui_pygame.py
"""

import pygame
import sys
import math
import random
import threading
import numpy as np
from collections import deque

# ──────────────────────────────────────────────────────────────────────────────
# Try to import the real project modules; fall back to inline stubs.
# ──────────────────────────────────────────────────────────────────────────────
try:
    from simulation.traffic_simulation import TrafficSimulation  # type: ignore
    from pso import run_pso                                       # type: ignore
    _USE_REAL = True
except Exception:
    _USE_REAL = False


# ══════════════════════════════════════════════════════════════════════════════
# INLINE SIMULATION (mirrors traffic_simulation.py / intersection.py / vehicle.py)
# ══════════════════════════════════════════════════════════════════════════════

NUM_INT      = 6
CYCLE_LEN    = 60
SIM_STEPS    = 500
SPAWN_PROB   = 0.7
PSO_POP      = 20
PSO_GENS     = 15


class _Vehicle:
    def __init__(self, start: int, dest: int):
        self.start       = start
        self.dest        = dest
        self.wait_time   = 0
        self.current_int = start
        self.done        = False


class _Intersection:
    def __init__(self, green_time=30, yellow_time=5, cycle_length=CYCLE_LEN):
        self.green_time  = green_time
        self.yellow_time = yellow_time
        self.red_time    = cycle_length - green_time - yellow_time
        self.cycle       = cycle_length
        self.time        = 0
        self.queue: list[_Vehicle] = []

    @property
    def phase(self) -> str:
        t = self.time % self.cycle
        if t < self.green_time:
            return "green"
        if t < self.green_time + self.yellow_time:
            return "yellow"
        return "red"

    @property
    def phase_remaining(self) -> int:
        t = self.time % self.cycle
        if t < self.green_time:
            return self.green_time - t
        if t < self.green_time + self.yellow_time:
            return self.green_time + self.yellow_time - t
        return self.cycle - t

    def update(self):
        self.time += 1
        for v in self.queue:
            v.wait_time += 1

    def release_vehicles(self) -> list[_Vehicle]:
        if self.phase != "green" or not self.queue:
            return []
        released = self.queue[:2]
        self.queue = self.queue[2:]
        return released

    def enqueue_vehicle(self, v: _Vehicle):
        self.queue.append(v)


def _run_sim(timings, steps=SIM_STEPS, spawn_prob=SPAWN_PROB):
    ints = [
        _Intersection(
            green_time=max(10, min(55, int(round(t)))),
            yellow_time=5,
            cycle_length=CYCLE_LEN,
        )
        for t in timings
    ]
    vehicles: list[_Vehicle] = []

    for _ in range(steps):
        if random.random() < spawn_prob:
            v = _Vehicle(0, NUM_INT - 1)
            vehicles.append(v)
            ints[0].enqueue_vehicle(v)

        for idx, inter in enumerate(ints):
            inter.update()
            released = inter.release_vehicles()
            for v in released:
                if idx + 1 < len(ints):
                    v.current_int = idx + 1
                    ints[idx + 1].enqueue_vehicle(v)
                else:
                    v.done = True

    avg_wait   = float(np.mean([v.wait_time for v in vehicles])) if vehicles else 0.0
    congestion = sum(len(i.queue) for i in ints)
    return avg_wait, congestion


def _fitness(avg_wait, cong):
    return avg_wait * 0.7 + cong * 2.0


def _repair(pos):
    return np.clip(np.round(pos), 10, 55)


def _run_pso(progress_cb=None):
    """Minimal PSO returning (best_timings, fitness, log).
    log = list of {"gen": int, "min": float, "avg": float}
    """
    dims    = NUM_INT
    c1 = c2 = 1.5
    v_min, v_max = -8.0, 8.0

    rng = np.random.default_rng(42)

    # particles: list of dicts
    particles = []
    for _ in range(PSO_POP):
        pos = _repair(rng.integers(10, 61, dims).astype(float))
        vel = rng.uniform(v_min, v_max, dims)
        particles.append({"pos": pos.copy(), "vel": vel,
                          "pbest": pos.copy(), "pbest_fit": np.inf})

    gbest     = None
    gbest_fit = np.inf
    log       = []

    for gen in range(PSO_GENS):
        W = 0.9 - (gen / PSO_GENS) * 0.5
        fits = []
        for p in particles:
            aw, cg = _run_sim(p["pos"], steps=120)
            f      = _fitness(aw, cg)
            fits.append(f)
            if f < p["pbest_fit"]:
                p["pbest_fit"] = f
                p["pbest"]     = p["pos"].copy()
            if f < gbest_fit:
                gbest_fit = f
                gbest     = p["pos"].copy()

        log.append({"gen": gen, "min": float(np.min(fits)), "avg": float(np.mean(fits))})

        for p in particles:
            r1, r2  = rng.random(dims), rng.random(dims)
            p["vel"] = np.clip(
                W * p["vel"]
                + c1 * r1 * (p["pbest"] - p["pos"])
                + c2 * r2 * (gbest     - p["pos"]),
                v_min, v_max,
            )
            p["pos"] = _repair(p["pos"] + p["vel"])

        if progress_cb:
            progress_cb(gen + 1, PSO_GENS, gbest_fit, log[-1]["avg"])

    best_timings = [int(x) for x in gbest]
    return best_timings, gbest_fit, log


# ══════════════════════════════════════════════════════════════════════════════
# COLOUR PALETTE
# ══════════════════════════════════════════════════════════════════════════════
BG          = (15,  17,  26)
ROAD_DARK   = (28,  32,  45)
ROAD_MID    = (38,  43,  58)
ROAD_LINE   = (200, 200, 140, 60)
PANEL_BG    = (22,  26,  38)
PANEL_LINE  = (50,  56,  72)
TEXT_PRI    = (230, 230, 230)
TEXT_SEC    = (140, 145, 160)
TEXT_ACC    = (100, 180, 255)
GREEN_C     = ( 34, 197,  94)
YELLOW_C    = (245, 158,  11)
RED_C       = (239,  68,  68)
CAR_COLORS  = [
    (255,  80,  80), ( 80, 160, 255), (255, 200,  60),
    ( 80, 220, 140), (200,  80, 255), (255, 140,  40),
    ( 60, 220, 220), (255, 100, 160),
]
CHART_MIN   = ( 55, 138, 221)
CHART_AVG   = (151, 196,  89)
CHART_BASE  = (239, 159,  39)

# ══════════════════════════════════════════════════════════════════════════════
# ANIMATED CAR
# ══════════════════════════════════════════════════════════════════════════════
class AnimCar:
    def __init__(self, color, lane_y):
        self.color   = color
        self.x       = -30.0
        self.y       = float(lane_y)
        self.tx      = 0.0          # target x (next intersection)
        self.ty      = float(lane_y)
        self.queued  = False
        self.done    = False
        self.inter_i = 0
        self.wait    = 0
        self.speed   = random.uniform(1.8, 2.8)
        self.length  = random.randint(22, 30)
        self.width   = 12
        # headlight flicker
        self._hl_t   = random.uniform(0, 6.28)

    def update(self):
        self._hl_t += 0.05
        if self.done:
            self.x += self.speed * 1.5
            return
        if self.queued:
            self.wait += 1
            return
        dx = self.tx - self.x
        if abs(dx) > 1:
            self.x += math.copysign(min(self.speed, abs(dx)), dx)
            dy = self.ty - self.y
            if abs(dy) > 0.5:
                self.y += math.copysign(min(0.5, abs(dy)), dy)

    def draw(self, surf):
        if self.x > surf.get_width() + 40:
            return
        cx, cy = int(self.x), int(self.y)
        hw, hh = self.length // 2, self.width // 2

        # Shadow
        shadow = pygame.Surface((self.length + 4, self.width + 2), pygame.SRCALPHA)
        shadow.fill((0, 0, 0, 60))
        surf.blit(shadow, (cx - hw - 2, cy - hh + hh + 2))

        # Body
        body_rect = pygame.Rect(cx - hw, cy - hh, self.length, self.width)
        pygame.draw.rect(surf, self.color, body_rect, border_radius=4)

        # Roof
        roof_w = int(self.length * 0.55)
        roof_h = self.width - 3
        roof_rect = pygame.Rect(cx - roof_w // 2, cy - hh + 1, roof_w, roof_h)
        darker = tuple(max(0, c - 30) for c in self.color)
        pygame.draw.rect(surf, darker, roof_rect, border_radius=3)

        # Windshields
        win_col = (160, 220, 255, 180)
        win_s = pygame.Surface((8, 6), pygame.SRCALPHA)
        win_s.fill(win_col)
        surf.blit(win_s, (cx + roof_w // 2 - 10, cy - hh + 2))
        surf.blit(win_s, (cx - roof_w // 2 + 2, cy - hh + 2))

        # Headlights
        hl_bright = int(200 + 40 * math.sin(self._hl_t))
        hl_col = (hl_bright, hl_bright, 180)
        pygame.draw.circle(surf, hl_col, (cx + hw - 2, cy - hh + 3), 2)
        pygame.draw.circle(surf, hl_col, (cx + hw - 2, cy + hh - 3), 2)

        # Taillights
        tl_col = (200, 30, 30)
        pygame.draw.circle(surf, tl_col, (cx - hw + 2, cy - hh + 3), 2)
        pygame.draw.circle(surf, tl_col, (cx - hw + 2, cy + hh - 3), 2)


# ══════════════════════════════════════════════════════════════════════════════
# TRAFFIC LIGHT WIDGET
# ══════════════════════════════════════════════════════════════════════════════
def draw_traffic_light(surf, cx, top_y, phase, remaining, green_time, font_sm):
    pole_w, pole_h = 4, 38
    box_w,  box_h  = 22, 62
    bx = cx - box_w // 2
    by = top_y

    # Pole
    pygame.draw.rect(surf, (90, 90, 100),
                     (cx - pole_w // 2, by + box_h, pole_w, pole_h))

    # Housing
    pygame.draw.rect(surf, (30, 32, 42), (bx, by, box_w, box_h), border_radius=5)
    pygame.draw.rect(surf, (60, 65, 80), (bx, by, box_w, box_h), 1, border_radius=5)

    colors = [RED_C, YELLOW_C, GREEN_C]
    active = {"red": 0, "yellow": 1, "green": 2}[phase]
    for i, col in enumerate(colors):
        cy_l = by + 10 + i * 18
        dim  = tuple(c // 4 for c in col)
        pygame.draw.circle(surf, dim if i != active else col, (cx, cy_l), 7)
        if i == active:
            glow = pygame.Surface((30, 30), pygame.SRCALPHA)
            pygame.draw.circle(glow, (*col, 60), (15, 15), 14)
            surf.blit(glow, (cx - 15, cy_l - 15))

    # Countdown
    txt = font_sm.render(f"{remaining}s", True, TEXT_SEC)
    surf.blit(txt, (cx - txt.get_width() // 2, by - 16))

    # Green time bar (below pole)
    bar_y = by + box_h + pole_h + 4
    bar_w = 50
    pygame.draw.rect(surf, (40, 44, 58), (cx - bar_w // 2, bar_y, bar_w, 5), border_radius=2)
    fill = int(bar_w * green_time / CYCLE_LEN)
    pygame.draw.rect(surf, GREEN_C, (cx - bar_w // 2, bar_y, fill, 5), border_radius=2)
    lbl = font_sm.render(f"{green_time}s", True, TEXT_ACC)
    surf.blit(lbl, (cx - lbl.get_width() // 2, bar_y + 7))


# ══════════════════════════════════════════════════════════════════════════════
# CHART DRAWER
# ══════════════════════════════════════════════════════════════════════════════
def draw_chart(surf, rect, log, baseline_fit, font_sm):
    pygame.draw.rect(surf, PANEL_BG, rect, border_radius=6)
    pygame.draw.rect(surf, PANEL_LINE, rect, 1, border_radius=6)

    if not log:
        msg = font_sm.render("Run PSO to see fitness chart", True, TEXT_SEC)
        surf.blit(msg, (rect.x + rect.w // 2 - msg.get_width() // 2,
                        rect.y + rect.h // 2 - msg.get_height() // 2))
        return

    pad = 28
    cw  = rect.w - pad * 2
    ch  = rect.h - pad * 2 - 10
    ox  = rect.x + pad
    oy  = rect.y + pad + 10

    all_vals = [d["min"] for d in log] + [d["avg"] for d in log]
    if baseline_fit:
        all_vals.append(baseline_fit)
    lo, hi = min(all_vals) * 0.88, max(all_vals) * 1.08
    rng    = hi - lo or 1

    def sx(i): return ox + int(i / (len(log) - 1 or 1) * cw)
    def sy(v): return oy + ch - int((v - lo) / rng * ch)

    # Grid
    for gv in np.linspace(lo, hi, 4):
        gy = sy(gv)
        pygame.draw.line(surf, PANEL_LINE, (ox, gy), (ox + cw, gy), 1)
        lbl = font_sm.render(f"{gv:.0f}", True, TEXT_SEC)
        surf.blit(lbl, (ox - lbl.get_width() - 3, gy - 6))

    # Baseline
    if baseline_fit:
        by_ = sy(baseline_fit)
        pygame.draw.line(surf, CHART_BASE, (ox, by_), (ox + cw, by_), 1)
        bl  = font_sm.render("baseline", True, CHART_BASE)
        surf.blit(bl, (ox + 4, by_ - 13))

    # Avg line
    if len(log) >= 2:
        pts = [(sx(i), sy(d["avg"])) for i, d in enumerate(log)]
        pygame.draw.lines(surf, CHART_AVG, False, pts, 2)

    # Min line + dots
    if len(log) >= 2:
        pts = [(sx(i), sy(d["min"])) for i, d in enumerate(log)]
        pygame.draw.lines(surf, CHART_MIN, False, pts, 2)
    for i, d in enumerate(log):
        pygame.draw.circle(surf, CHART_MIN, (sx(i), sy(d["min"])), 3)

    # Legend
    for col, lbl in [(CHART_MIN, "Min"), (CHART_AVG, "Avg"), (CHART_BASE, "Baseline")]:
        pass  # drawn inline above


# ══════════════════════════════════════════════════════════════════════════════
# BUTTON
# ══════════════════════════════════════════════════════════════════════════════
class Button:
    def __init__(self, rect, label, accent=False):
        self.rect   = pygame.Rect(rect)
        self.label  = label
        self.accent = accent
        self.hover  = False
        self.active = False

    def draw(self, surf, font):
        if self.active:
            col = TEXT_ACC
            bg  = (20, 50, 80)
        elif self.hover:
            bg, col = (40, 46, 62), TEXT_PRI
        else:
            bg, col = (30, 35, 50), TEXT_SEC
        pygame.draw.rect(surf, bg,         self.rect, border_radius=6)
        pygame.draw.rect(surf, PANEL_LINE, self.rect, 1, border_radius=6)
        txt = font.render(self.label, True, col)
        surf.blit(txt, (self.rect.centerx - txt.get_width() // 2,
                        self.rect.centery - txt.get_height() // 2))

    def handle(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False


# ══════════════════════════════════════════════════════════════════════════════
# SLIDER
# ══════════════════════════════════════════════════════════════════════════════
class Slider:
    def __init__(self, x, y, w, lo, hi, val, label, fmt="{:.0f}"):
        self.rect   = pygame.Rect(x, y, w, 16)
        self.lo, self.hi = lo, hi
        self.val    = val
        self.label  = label
        self.fmt    = fmt
        self._drag  = False

    @property
    def frac(self):
        return (self.val - self.lo) / (self.hi - self.lo)

    def thumb_x(self):
        return int(self.rect.x + self.frac * self.rect.w)

    def draw(self, surf, font):
        # Track
        ty = self.rect.centery
        pygame.draw.line(surf, PANEL_LINE,
                         (self.rect.x, ty), (self.rect.right, ty), 3)
        pygame.draw.line(surf, TEXT_ACC,
                         (self.rect.x, ty), (self.thumb_x(), ty), 3)
        # Thumb
        pygame.draw.circle(surf, TEXT_ACC, (self.thumb_x(), ty), 7)
        pygame.draw.circle(surf, TEXT_PRI, (self.thumb_x(), ty), 5)
        # Labels
        lbl = font.render(self.label, True, TEXT_SEC)
        surf.blit(lbl, (self.rect.x, self.rect.y - 18))
        val_s = font.render(self.fmt.format(self.val), True, TEXT_ACC)
        surf.blit(val_s, (self.rect.right - val_s.get_width(), self.rect.y - 18))

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            tx = self.thumb_x()
            ty = self.rect.centery
            if math.hypot(event.pos[0] - tx, event.pos[1] - ty) < 12:
                self._drag = True
        if event.type == pygame.MOUSEBUTTONUP:
            self._drag = False
        if event.type == pygame.MOUSEMOTION and self._drag:
            frac = (event.pos[0] - self.rect.x) / self.rect.w
            self.val = round(self.lo + max(0, min(1, frac)) * (self.hi - self.lo), 2)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
def main():
    pygame.init()
    pygame.display.set_caption("Traffic Signal Optimisation – PSO  |  AI420")
    W, H = 1100, 700
    screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
    clock  = pygame.time.Clock()

    font_lg = pygame.font.SysFont("segoeui", 20, bold=True)
    font_md = pygame.font.SysFont("segoeui", 16)
    font_sm = pygame.font.SysFont("segoeui", 13)
    font_xs = pygame.font.SysFont("segoeui", 11)

    # ── Layout constants (recalculated on resize) ──
    ROAD_Y     = 260        # centre of road
    ROAD_H     = 80
    LEFT_PAD   = 20
    PANEL_W    = 280
    SIM_W      = W - PANEL_W - LEFT_PAD * 3

    inter_xs   = [
        int(LEFT_PAD + 60 + i * (SIM_W - 120) / (NUM_INT - 1))
        for i in range(NUM_INT)
    ]

    # ── State ──
    mode            = "baseline"   # "baseline" | "pso" | "compare"
    base_timings    = [30] * NUM_INT
    pso_timings     = None
    pso_log         = []
    baseline_fit    = None
    pso_fit         = None
    pso_running     = False
    pso_progress    = [0, PSO_GENS, 0.0, 0.0]   # gen, total, best, avg

    # Live intersection state (phase / time / queue)
    live_ints = [_Intersection(30, 5, CYCLE_LEN) for _ in range(NUM_INT)]
    live_cars: list[AnimCar] = []

    spawn_tick  = 0
    stats_wait  = deque(maxlen=120)
    stats_cong  = deque(maxlen=120)

    # ── Buttons ──
    bx = W - PANEL_W - LEFT_PAD + 10
    btn_base    = Button((bx, 90,  120, 32), "Baseline",  True)
    btn_pso     = Button((bx, 130, 120, 32), "Run PSO",   True)
    btn_compare = Button((bx, 170, 120, 32), "Compare",   True)
    btn_reset   = Button((bx + 130, 90, 80, 32), "Reset")
    btn_base.active = True

    # ── Sliders ──
    sx0 = bx
    sl_spawn = Slider(sx0, 230, PANEL_W - 20, 0.1, 1.0, 0.7, "Spawn rate", "{:.1f}")
    sl_speed = Slider(sx0, 280, PANEL_W - 20, 1,   8,   3,   "Sim speed",  "×{:.0f}")

    # ── Chart rect ──
    chart_rect = pygame.Rect(bx, 390, PANEL_W - 10, 200)

    # ── Helpers ──
    def set_mode(m):
        nonlocal mode, live_ints, live_cars, spawn_tick
        mode = m
        btn_base.active    = (m == "baseline")
        btn_pso.active     = (m == "pso")
        btn_compare.active = (m == "compare")
        timings = base_timings if m == "baseline" else (pso_timings or base_timings)
        live_ints  = [
            _Intersection(max(10, min(55, int(round(t)))), 5, CYCLE_LEN)
            for t in timings
        ]
        live_cars  = []
        spawn_tick = 0

    def do_reset():
        nonlocal pso_timings, pso_log, baseline_fit, pso_fit, pso_running, pso_progress
        pso_timings   = None
        pso_log       = []
        baseline_fit  = None
        pso_fit       = None
        pso_running   = False
        pso_progress  = [0, PSO_GENS, 0.0, 0.0]
        stats_wait.clear(); stats_cong.clear()
        set_mode("baseline")

    def launch_pso():
        nonlocal pso_running, pso_timings, pso_log, pso_fit

        if pso_running:
            return
        pso_running = True
        pso_log.clear()

        # Compute baseline fitness first
        nonlocal baseline_fit
        if baseline_fit is None:
            aw, cg    = _run_sim(base_timings)
            baseline_fit = _fitness(aw, cg)

        def _cb(gen, total, best, avg):
            pso_progress[:] = [gen, total, best, avg]

        def _worker():
            nonlocal pso_timings, pso_fit, pso_running
            timings, fit, log = _run_pso(_cb)
            pso_timings  = timings
            pso_fit      = fit
            pso_log[:]   = log
            pso_running  = False

        threading.Thread(target=_worker, daemon=True).start()

    def spawn_car(inter_i=0):
        col  = random.choice(CAR_COLORS)
        lane = ROAD_Y + random.choice([-14, -5, 5, 14])
        car  = AnimCar(col, lane)
        car.x      = inter_xs[0] - 120.0
        car.tx     = float(inter_xs[inter_i]) - 10
        car.ty     = float(lane)
        car.inter_i = inter_i
        live_ints[inter_i].enqueue_vehicle(_Vehicle(inter_i, NUM_INT - 1))
        live_cars.append(car)

    def update_sim():
        nonlocal spawn_tick
        speed = int(sl_speed.val)

        for _ in range(speed):
            spawn_tick += 1
            if spawn_tick % max(1, int(10 / sl_spawn.val)) == 0:
                spawn_car(0)

            for idx, inter in enumerate(live_ints):
                inter.update()
                released = inter.release_vehicles()
                if released and idx + 1 < NUM_INT:
                    for _ in released:
                        # Move the first queued visual car at this inter forward
                        cands = [c for c in live_cars
                                 if c.inter_i == idx and c.queued and not c.done]
                        if cands:
                            c         = cands[0]
                            c.inter_i = idx + 1
                            c.queued  = False
                            c.tx      = float(inter_xs[idx + 1]) - 10
                            c.ty      = c.y
                elif released and idx + 1 == NUM_INT:
                    cands = [c for c in live_cars
                             if c.inter_i == idx and c.queued and not c.done]
                    for c in cands[:len(released)]:
                        c.done   = True
                        c.queued = False

            # Park cars at their intersection
            for c in live_cars:
                if c.done:
                    continue
                dist = abs(c.x - c.tx)
                if dist < 4 and not c.queued:
                    c.queued = True

                # Check if light turned green → release queue visually
                inter = live_ints[c.inter_i]
                if c.queued and inter.phase == "green":
                    if inter_xs[c.inter_i + 1] if c.inter_i + 1 < NUM_INT else None:
                        pass   # handled above

            # Update positions
            for c in live_cars:
                c.update()

        # Trim old cars
        if len(live_cars) > 300:
            live_cars[:] = [c for c in live_cars if not c.done][-280:]

        # Stats
        queued = sum(1 for c in live_cars if c.queued and not c.done)
        waits  = [c.wait for c in live_cars if c.queued and not c.done]
        stats_wait.append(float(np.mean(waits)) if waits else 0.0)
        stats_cong.append(float(queued))

    # ── Draw helpers ──
    def draw_road(surf):
        rw = SIM_W + LEFT_PAD * 2
        # Base road
        pygame.draw.rect(surf, ROAD_DARK, (0, ROAD_Y - ROAD_H // 2, rw, ROAD_H))
        # Kerbs
        pygame.draw.rect(surf, ROAD_MID,  (0, ROAD_Y - ROAD_H // 2,     rw, 6))
        pygame.draw.rect(surf, ROAD_MID,  (0, ROAD_Y + ROAD_H // 2 - 6, rw, 6))
        # Centre dashes (use Surface for alpha)
        for dx in range(0, rw, 48):
            dash = pygame.Surface((28, 2), pygame.SRCALPHA)
            dash.fill(ROAD_LINE)
            surf.blit(dash, (dx, ROAD_Y - 1))
        # Lane markers
        for dx in range(0, rw, 48):
            dash = pygame.Surface((28, 1), pygame.SRCALPHA)
            dash.fill((255, 255, 200, 25))
            surf.blit(dash, (dx, ROAD_Y - 20))
            surf.blit(dash, (dx, ROAD_Y + 20))

    def draw_intersections(surf):
        for i, ix in enumerate(inter_xs):
            inter = live_ints[i]
            # Stop line
            pygame.draw.line(surf, (180, 180, 120, 80),
                             (ix - 2, ROAD_Y - ROAD_H // 2),
                             (ix - 2, ROAD_Y + ROAD_H // 2), 1)
            draw_traffic_light(
                surf, ix,
                ROAD_Y - ROAD_H // 2 - 80,
                inter.phase,
                inter.phase_remaining,
                inter.green_time,
                font_xs,
            )
            # Queue count
            if inter.queue:
                q_s = font_xs.render(f"Q:{len(inter.queue)}", True, YELLOW_C)
                surf.blit(q_s, (ix - q_s.get_width() // 2,
                                ROAD_Y + ROAD_H // 2 + 12))

    def draw_panel(surf):
        px = W - PANEL_W - LEFT_PAD
        pygame.draw.rect(surf, PANEL_BG,  (px, 0, PANEL_W + LEFT_PAD, H))
        pygame.draw.line(surf, PANEL_LINE, (px, 0), (px, H), 1)

        # Title
        t = font_lg.render("PSO Traffic Optimiser", True, TEXT_PRI)
        surf.blit(t, (px + 10, 14))
        s = font_xs.render("AI420 – Spring 2026", True, TEXT_SEC)
        surf.blit(s, (px + 10, 40))

        # Mode buttons
        btn_base.draw(surf, font_md)
        btn_pso.draw(surf, font_md)
        btn_compare.draw(surf, font_md)
        btn_reset.draw(surf, font_md)

        # PSO progress bar
        if pso_running or pso_log:
            bx_ = px + 10
            gen = pso_progress[0]
            bar_frac = gen / PSO_GENS
            pygame.draw.rect(surf, PANEL_LINE, (bx_, 195, PANEL_W - 20, 8), border_radius=4)
            pygame.draw.rect(surf, TEXT_ACC,
                             (bx_, 195, int((PANEL_W - 20) * bar_frac), 8), border_radius=4)
            prog_s = font_xs.render(
                f"Gen {gen}/{PSO_GENS}  best: {pso_progress[2]:.1f}" if pso_running
                else f"Done – best fitness: {pso_fit:.1f}" if pso_fit else "",
                True, TEXT_SEC,
            )
            surf.blit(prog_s, (bx_, 207))

        # Sliders
        sl_spawn.draw(surf, font_sm)
        sl_speed.draw(surf, font_sm)

        # Stats
        sy0 = 310
        def stat(label, val, col=TEXT_PRI):
            nonlocal sy0
            l = font_sm.render(label, True, TEXT_SEC)
            v = font_md.render(str(val), True, col)
            surf.blit(l, (px + 10, sy0))
            surf.blit(v, (px + PANEL_W - v.get_width() - 18, sy0))
            sy0 += 22

        aw   = f"{stats_wait[-1]:.1f}s" if stats_wait else "—"
        cg   = f"{int(stats_cong[-1])}"  if stats_cong else "—"
        bl   = f"{baseline_fit:.1f}"      if baseline_fit else "—"
        po   = f"{pso_fit:.1f}"           if pso_fit      else "—"
        imp  = (f"{(baseline_fit - pso_fit) / baseline_fit * 100:.1f}%"
                if baseline_fit and pso_fit else "—")

        stat("Mode:", {"baseline": "Baseline", "pso": "PSO", "compare": "Compare"}[mode])
        stat("Avg wait:", aw)
        stat("Congestion:", cg)
        stat("Baseline fitness:", bl, TEXT_SEC)
        stat("PSO fitness:", po, GREEN_C if pso_fit else TEXT_PRI)
        stat("Improvement:", imp, GREEN_C if pso_fit else TEXT_PRI)

        # Timing table header
        th = font_xs.render("INT   BASE   PSO", True, TEXT_SEC)
        surf.blit(th, (px + 10, sy0)); sy0 += 16
        pygame.draw.line(surf, PANEL_LINE, (px + 10, sy0), (px + PANEL_W - 10, sy0), 1)
        sy0 += 4
        for i in range(NUM_INT):
            base_g = 30
            pso_g  = pso_timings[i] if pso_timings else "—"
            row    = f"  {i+1}    {base_g}s     {pso_g}s" if pso_timings else f"  {i+1}    {base_g}s     —"
            col    = TEXT_ACC if mode in ("pso", "compare") and pso_timings else TEXT_PRI
            rs     = font_xs.render(row, True, col)
            surf.blit(rs, (px + 10, sy0)); sy0 += 14

        # Chart
        draw_chart(surf, chart_rect, pso_log, baseline_fit, font_xs)

        # Legend chips
        ly = chart_rect.bottom + 6
        for col, lbl in [(CHART_MIN, "Min"), (CHART_AVG, "Avg"), (CHART_BASE, "Baseline")]:
            pygame.draw.circle(surf, col, (px + 14, ly + 5), 4)
            t = font_xs.render(lbl, True, TEXT_SEC)
            surf.blit(t, (px + 22, ly))
            px += t.get_width() + 30

    def draw_bg_buildings(surf):
        """Simple silhouette skyline behind road."""
        bldgs = [(30,H-ROAD_Y+40,40,80),(90,H-ROAD_Y+40,30,110),
                 (160,H-ROAD_Y+40,50,60),(260,H-ROAD_Y+40,35,95),
                 (370,H-ROAD_Y+40,45,70),(470,H-ROAD_Y+40,30,115),
                 (560,H-ROAD_Y+40,55,55),(660,H-ROAD_Y+40,40,90),
                 (750,H-ROAD_Y+40,50,75)]
        for bx_, by_, bw, bh in bldgs:
            col = (24, 28, 40)
            pygame.draw.rect(surf, col,
                             (bx_, ROAD_Y - ROAD_H // 2 - bh, bw, bh))
            # Windows
            for wy in range(ROAD_Y - ROAD_H // 2 - bh + 8, ROAD_Y - ROAD_H // 2 - 4, 14):
                for wx in range(bx_ + 5, bx_ + bw - 5, 10):
                    win_col = random.choice([(50,55,70),(70,65,40),(30,40,60)])
                    pygame.draw.rect(surf, win_col, (wx, wy, 5, 7))

    # ──────────────────────────────────────────────────────────────────────────
    # MAIN LOOP
    # ──────────────────────────────────────────────────────────────────────────
    running = True
    while running:
        dt = clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            if event.type == pygame.VIDEORESIZE:
                W, H = event.w, event.h
                screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
                PANEL_W = 280
                SIM_W   = W - PANEL_W - LEFT_PAD * 3
                inter_xs[:] = [
                    int(LEFT_PAD + 60 + i * (SIM_W - 120) / (NUM_INT - 1))
                    for i in range(NUM_INT)
                ]
                bx = W - PANEL_W - LEFT_PAD + 10
                btn_base.rect.x    = bx
                btn_pso.rect.x     = bx
                btn_compare.rect.x = bx
                btn_reset.rect.x   = bx + 130
                sl_spawn.rect.x    = bx
                sl_speed.rect.x    = bx
                chart_rect.x       = bx
                chart_rect.y       = 390

            if btn_base.handle(event):
                set_mode("baseline")
            if btn_pso.handle(event):
                if not pso_running:
                    set_mode("pso")
                    launch_pso()
            if btn_compare.handle(event):
                if pso_timings:
                    set_mode("compare")
            if btn_reset.handle(event):
                do_reset()
            sl_spawn.handle(event)
            sl_speed.handle(event)

        # ── After PSO finishes, switch live ints to optimised timings ──
        if mode == "pso" and not pso_running and pso_timings:
            live_ints = [
                _Intersection(max(10, min(55, int(t))), 5, CYCLE_LEN)
                for t in pso_timings
            ]

        update_sim()

        # ── Render ──
        screen.fill(BG)
        draw_bg_buildings(screen)
        draw_road(screen)

        # Cars (back-to-front)
        for car in sorted(live_cars, key=lambda c: c.y):
            car.draw(screen)

        draw_intersections(screen)
        draw_panel(screen)

        # HUD title bar
        mode_s = {"baseline": "BASELINE  –  30s uniform green",
                  "pso":      "PSO OPTIMISED" + (" – running…" if pso_running else f"  –  fitness {pso_fit:.1f}" if pso_fit else ""),
                  "compare":  f"COMPARE  –  Baseline vs PSO  (↑ {((baseline_fit-pso_fit)/baseline_fit*100):.1f}% better)" if baseline_fit and pso_fit else "COMPARE"}[mode]
        hud = font_md.render(mode_s, True, TEXT_ACC)
        screen.blit(hud, (LEFT_PAD, H - 26))
        fps_s = font_xs.render(f"FPS {clock.get_fps():.0f}", True, TEXT_SEC)
        screen.blit(fps_s, (LEFT_PAD, H - 44))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
