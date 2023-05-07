from src.car import Car


class CarQueue:
    # A class variable to keep track of all car queues.
    all_car_queues = []

    def __init__(self, max_capacity, id):
        CarQueue.all_car_queues.append(self)
        self.id = id
        self.cars = []
        self.capacity = max_capacity
        self.num_of_cars = len(self.cars)
        self.time_inactive = 0
        self.bids = {}
        self.total_fee = 0

    def __str__(self):
        # Create list of all IDs of cars in the queue
        car_ids = []
        for car in self.cars:
            car_ids.append(car.id)

        return f'Car Queue (ID: {self.id}) contains cars with IDs: {car_ids}'

    def queue_length(self):
        return len(self.cars)

    def add_car(self, car):
        if self.has_capacity():
            self.cars.append(car)
        else:
            print("Car Queue is full, cannot add car")
            return -1

    def get_destination(self):
        # For now, the car is not removed. We first need to check if the new queue has capacity.
        car = self.cars[0]
        destination = car.update_destination_queue()
        return destination

    def remove_first_car(self):
        # Return the first car from the queue
        if self.is_empty():
            print("ERROR: Cannot remove car from empty queue")
            return -1
        return self.cars.pop(0)

    def has_capacity(self):
        if self.queue_length() < self.capacity:
            return True
        else:
            return False
        
    def get_num_of_free_spots(self):
        return self.capacity - self.queue_length()

    def is_empty(self):
        if self.queue_length() > 0:
            return False
        else:
            return True

    def collect_bids(self):
        bids = {}
        # A dictionary is used so that we know which car submitted which bid, as well as the waiting time.
        for car in self.cars:
            car_id, bid, time_inactive = car.submit_bid()
            bids[car_id] = [bid, time_inactive]
        return bids

    def set_auction_fee(self, fee):
        self.total_fee = fee

    def win_auction(self):
        # If a queue wins an auction:
        # 1. The winning bid is paid by the cars in the queue.
        # 2. The inactivity time is reset for the cars in the queue.
        # 3. The inactivity time is reset for the queue itself.
        # - NOTE: The actual winning car is moved by the grid itself.
        # First, the bid must be paid
        total_amount_paid = 0  # NOTE: Remove this if it works DEBUG ONLY!!!
        queue_car_ids = []  # This holds the IDs of all cars in the queue
        queue_bids = []  # This holds the bids of all cars in the queue, in the same order as the IDs
        total_submitted_bid = 0  # This is the sum of the bids of all cars in the queue
        
        for bid in self.bids.items():
            queue_car_ids.append(bid[0])
            queue_bids.append(bid[1][0])
            total_submitted_bid += bid[1][0]

        for i in range(len(queue_car_ids)):
            # The winning bid is divided proportionally depending on the individual bids of the cars in the queue.
            individual_price = self.total_fee * \
                queue_bids[i] / total_submitted_bid
            self.cars[i].pay_bid(individual_price)
            total_amount_paid += individual_price

        print("In total, the cars in the queue paid {} to exit the intersection. The total amount to be paid was: {}".format(
            total_amount_paid, self.total_fee))

        # Second, the inactivity time must be reset for all cars in the queue.
        for car in self.cars:
            car.time_inactive = 0

        # Third, the inactivity time must be reset for the queue itself.
        self.time_inactive = 0

    def reset_bids(self):
        # Reset the bids for the car queue, so that the next auction can be run.
        self.bids = self.bids.clear()

    def ready_for_new_epoch(self):
        self.reset_bids()
        self.num_of_cars = self.queue_length()
        self.time_inactive += 1
        self.total_fee = 0
