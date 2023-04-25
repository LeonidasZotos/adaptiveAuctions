
class CarQueue:
    def __init__(self, max_capacity):
        self.cars = []
        self.capacity = max_capacity
        self.num_of_cars = len(self.cars)
        self.time_inactive = 0

    def __str__(self):
        return f'CarQueue(id={self.id}), contains cars with IDs: {self.cars}'

    def queue_length(self):
        return len(self.cars)

    def add_car(self, car):
        self.cars.append(car)

    def get_first_car(self):
        self.cars.pop(0)
        
    def has_capacity(self):
        if self.queue_length() < self.capacity:
            return True
        
    def collect_bids(self):
        # for each car in the queue, call submit_bid() and add the bid to a list
        bids = []
        for car in self.cars:
            bids.append(car.submit_bid())
        return bids
        