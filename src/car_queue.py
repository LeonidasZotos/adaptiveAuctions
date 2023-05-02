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

    def get_first_car_and_destination(self):
        # For now, the car is not removed. We first need to check if the new queue has capacity.
        car = self.cars[0]
        destination = car.update_destination_queue()
        return car, destination

    def has_capacity(self):
        if self.queue_length() < self.capacity:
            return True
        else:
            return False

    def collect_bids(self):
        bids = {}
        # A dictionary is used so that we know which car submitted which bid, as well as the waiting time.
        for car in self.cars:
            car_id, bid, waiting_time = car.submit_bid()
            bid[car_id] = (bid, waiting_time)
        return bids

    def win_auction(self, winning_bid):
        # This is executed by the car queue that won the auction. The winning bid is given by the mechanism, and can be the 2nd bid for example.
        # The waiting time is also reset for the cars in this queue.
        # TODO: Only commit actions if the new queue has capacity!!!

        # First, the bid must be paid
        total_amount_paid = 0  # TODO: Remove this if it works DEBUG ONLY!!!
        queue_car_ids = []  # This holds the IDs of all cars in the queue
        queue_bids = []  # This holds the bids of all cars in the queue, in the same order as the IDs
        total_submitted_bid = 0  # This is the sum of the bids of all cars in the queue
        for bid in self.bids.items:
            queue_car_ids.append(bid[0])
            queue_bids.append(bid[1][0])
            total_submitted_bid += bid[1][0]

        for i in range(len(queue_car_ids)):
            # The winning bid is divided proportionally depending on the individual bids of the cars in the queue.
            individual_price = winning_bid * \
                queue_bids[i] / total_submitted_bid
            self.cars[i].pay_bid(individual_price)
            total_amount_paid += individual_price

        print("In total, the cars in the queue paid {} to exit the intersection. The total amount to be paid was: {}".format(
            total_amount_paid, winning_bid))

        # Second, the waiting time must be reset, as the cars received priority
        for car in self.cars:
            car.waiting_time = 0

        # Third, the bids must be reset, so that the next auction can be run.
        self.reset_bids()

        # Lastly, the car that won the auction must be removed from the queue
        winning_car, destination = self.get_first_car_and_destination()
        print("Winning car with iD {} and destination {} got priority".format(
            winning_car.id, destination))

    def lose_auction(self):
        # This is executed by the car queues that lost the auction.
        # First, the waiting time must be increased for all cars in the queue.
        for car in self.cars:
            car.waiting_time += 1
        # Second, the bids must be reset, so that the next auction can be run.
        self.reset_bids()

    def reset_bids(self):
        # Reset the bids for the car queue, so that the next auction can be run.
        self.bids = self.bids.clear()
