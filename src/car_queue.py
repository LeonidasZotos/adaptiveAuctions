from src.car import Car


class CarQueue:
    # A class variable to keep track of all car queues.
    all_car_queues = []
    
    def __init__(self, max_capacity):
        CarQueue.all_car_queues.append(self)
        self.cars = []
        self.capacity = max_capacity
        self.num_of_cars = len(self.cars)
        self.time_inactive = 0

    def __str__(self):
        # Create list of all IDs of cars in the queue
        car_ids = []
        for car in self.cars:
            car_ids.append(car.id)
        
        return f'CarQueue contains cars with IDs: {car_ids}'

    def queue_length(self):
        return len(self.cars)

    def add_car(self, car):
        if self.has_capacity():
            self.cars.append(car)
        else: 
            print("CarQueue is full, cannot add car")
            return -1

    def get_first_car(self):
        self.cars.pop(0)

    def has_capacity(self):
        if self.queue_length() < self.capacity:
            return True

    def collect_bids(self):
        bids = {}
        # A dictionary is used so that we know which car submitted which bid.
        for car in self.cars:
            car_id, bid = car.submit_bid()
            bid[car_id] = bid
        return bids
