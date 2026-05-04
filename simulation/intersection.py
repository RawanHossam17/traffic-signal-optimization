class Intersection:
    def __init__(self, green_time, yellow_time=5, cycle_length=60):
        self.green_time = green_time
        self.yellow_time = yellow_time
        self.red_time = cycle_length - (green_time + yellow_time)
        self.cycle_length = cycle_length
        self.time = 0
        self.queue = []
        self.wait_time_counter = 0

    def update(self):
        self.time = (self.time + 1) % self.cycle_length
        for vehicle in self.queue:
            vehicle.wait_time += 1

    def current_light(self):
        if self.time < self.green_time:
            return "GREEN"
        elif self.time < self.green_time + self.yellow_time:
            return "YELLOW"
        else:
            return "RED"

    def enqueue_vehicle(self, vehicle):
        vehicle.current_intersection = self
        self.queue.append(vehicle)

    def release_vehicles(self):
        light = self.current_light()
        released = []
        if light == "GREEN":
            release_count = min(2, len(self.queue))
            for _ in range(release_count):
                if self.queue:
                    v = self.queue.pop(0)
                    v.move()
                    released.append(v)
        elif light == "YELLOW":
            if self.queue:
                v = self.queue.pop(0)
                v.speed = max(0.5, v.speed * 0.5)
                v.move()
                released.append(v)
        return released