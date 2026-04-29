class Vehicle:
    def __init__(self, start, destination, speed=1):
        self.position = start
        self.destination = destination
        self.wait_time = 0
        self.speed = speed
        self.acceleration = 0.1
        self.deceleration = 0.2

    def move(self):
        self.speed += self.acceleration
        if self.speed > 2:  
            self.speed = 2
        if self.position < self.destination:
            self.position += self.speed
        else:
            self.position = self.destination

    def stop(self):
        self.speed -= self.deceleration
        if self.speed < 0:
            self.speed = 0
        self.wait_time += 1
