from .intersection import Intersection
from .vehicle import Vehicle
import numpy as np
import random

class TrafficSimulation:
    def __init__(self, num_intersections, cycle_length=60):
        self.intersections = [
            Intersection(green_time=30, yellow_time=5, cycle_length=cycle_length)
            for _ in range(num_intersections)
        ]
        self.vehicles = []
        self.cycle_length = cycle_length

    def add_vehicle(self, start=0, destination=None):
        if destination is None:
            destination = len(self.intersections) - 1
        v = Vehicle(start, destination)
        self.vehicles.append(v)
        self.intersections[start].enqueue_vehicle(v)

    def run(self, signal_timings, steps=500):
        self.vehicles = []

        for inter in self.intersections:
            inter.queue = []
            inter.time = 0
        for i, timing in enumerate(signal_timings):
            self.intersections[i].green_time = timing
            self.intersections[i].yellow_time = 5
            self.intersections[i].red_time = self.cycle_length - (timing + self.intersections[i].yellow_time)

        for step in range(steps):
            if random.random() < 0.7:
               self.add_vehicle()

            for idx, inter in enumerate(self.intersections):
                inter.update()
                released = inter.release_vehicles()
                if released and idx + 1 < len(self.intersections):
                    for v in released:
                        self.intersections[idx + 1].enqueue_vehicle(v)

        avg_wait = np.mean([v.wait_time for v in self.vehicles]) if self.vehicles else 0
        congestion = sum([len(inter.queue) for inter in self.intersections])
        return avg_wait, congestion
