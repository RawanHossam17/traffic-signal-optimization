from simulation.intersection import Intersection
from simulation.vehicle import Vehicle
import numpy as np


class TrafficSimulation:
    def __init__(self, num_intersections, cycle_length=60, traffic_level="2"):
        self.num_intersections = num_intersections
        self.cycle_length = cycle_length
        self.traffic_level = traffic_level
        self.reset()

    def reset(self):
        self.intersections = [
            Intersection(green_time=30, yellow_time=5, cycle_length=self.cycle_length)
            for _ in range(self.num_intersections)
        ]
        self.vehicles = []
        self.completed_vehicles = []
        self.total_wait_time = 0
        self.vehicle_counter = 0

    def get_arrival_rate(self):
        if self.traffic_level == "1":
            return 8
        elif self.traffic_level == "2":
            return 5
        else:
            return 3

    def add_vehicle(self, start=0, destination=None):
        if destination is None:
            destination = len(self.intersections) - 1
        if start >= destination:
            start = 0
            destination = self.num_intersections - 1
        v = Vehicle(start, destination)
        self.vehicles.append(v)
        self.intersections[start].enqueue_vehicle(v)

    def run(self, signal_timings, steps=100):
        self.reset()

        for i, timing in enumerate(signal_timings):
            if i < len(self.intersections):
                self.intersections[i].green_time = max(10, min(50, timing))
                self.intersections[i].red_time = max(5, self.cycle_length - (timing + 5))

        for step in range(steps):
            if step % self.get_arrival_rate() == 0:
                start = np.random.randint(0, max(1, self.num_intersections - 1))
                self.add_vehicle(start)

            vehicles_to_move = []
            for idx, inter in enumerate(self.intersections):
                inter.update()
                released = inter.release_vehicles()
                vehicles_to_move.extend([(idx, v) for v in released])

            for idx, v in vehicles_to_move:
                if idx + 1 < self.num_intersections:
                    self.intersections[idx + 1].enqueue_vehicle(v)
                else:
                    self.completed_vehicles.append(v)

        if self.vehicles:
            wait_times = [v.wait_time for v in self.vehicles if v.wait_time > 0]
            avg_wait = np.mean(wait_times) if wait_times else 0
        else:
            avg_wait = 0

        congestion = sum([len(inter.queue) for inter in self.intersections])
        return avg_wait, congestion