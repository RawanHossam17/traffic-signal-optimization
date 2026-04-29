class Intersection:
    def __init__(self, green_time, yellow_time=5, cycle_length=60):
        self.green_time = green_time
        self.yellow_time = yellow_time
        self.red_time = cycle_length - (green_time + yellow_time)
        self.cycle_length = cycle_length
        self.time = 0
        self.queue = []

    def update(self):
        self.time = (self.time + 1) % self.cycle_length

    def current_light(self):
        if self.time < self.green_time:
            return "GREEN"
        elif self.time < self.green_time + self.yellow_time:
            return "YELLOW"
        else:
            return "RED"

    def enqueue_vehicle(self, vehicle):
        self.queue.append(vehicle)

    def release_vehicles(self):
        light = self.current_light()
        released = []
        if light == "GREEN":
            for v in self.queue[:1]: 
                v.move()
                released.append(v)
            self.queue = self.queue[1:]
        elif light == "YELLOW":
            for v in self.queue[:1]:
                v.speed = max(0.5, v.speed * 0.5)
                v.move()
                released.append(v)
            self.queue = self.queue[1:]
        else:
            for v in self.queue:
                v.wait_time += 1
                v.stop()
        return released
