import random


class Car:
    # A class variable to keep track of all cars.
    all_cars = []

    def __init__(self, id, car_queue_id, grid_size):
        Car.all_cars.append(self)
        self.id = id
        # Randomly pick a destination intersection
        self.final_destination = self.set_final_destination(grid_size)
        # Set an initial balance
        self.balance = 0
        # Rush factor is random between 0 and 1
        self.rush_factor = random.random()
        self.submitted_bid = 0
        self.waiting_time = 0
        # location_id is the ID of the intersection and queue the car is currently in (e.g. 11N, for intersection (1,1), north car queue).
        self.car_queue_id = car_queue_id

    def __str__(self):
        return f'Car(id={self.id}), with destination: {self.final_destination}'

    def submit_bid(self):
        # For now, randomly submit a bid. TODO: Incorporate rush factor
        self.submitted_bid = random.randint(1, 20)
        # If there is not enough balance, bid entire balance.
        if self.submitted_bid > self.balance:
            self.submitted_bid = self.balance

        # Return a tuple of the car ID and the bid.
        # Car ID is included so that the intersection can keep track of which car submitted which bid.
        return self.id, self.submitted_bid, self.waiting_time

    def pay_bid(self, price):
        # Price is not necessarily the same as the bid (e.g. 2nd price auction)
        self.balance -= price
        if self.balance < 0:  # This should never happen, we the bid is limited by the balance
            print("ERROR: Car {} has negative balance".format(self.id))
        print("Car {} paid {} to exit intersection".format(self.id, price))

    def set_balance(self, balance):
        # Set the balance of the car to the given balance. Used for the wage distribution.
        self.balance = balance

    def set_final_destination(self, grid_size):
        # Randomly pick a destination intersection
        x = random.randint(0, grid_size - 1)
        y = random.randint(0, grid_size - 1)
        self.final_destination = str(x) + str(y)
        print("The final destination of car {} is {}".format(
            self.id, self.final_destination))

    def is_at_destination(self):
        # Evaluated whether the car is at its final destination.
        # Remove the last character from  car_queue_id (e.g. 11N -> 11).
        current_intersection = self.car_queue_id[:-1]
        if current_intersection == self.final_destination:
            return True
        else:
            return False

    def retrieve_intersection_exit(self):
        # Returns a queue ID: <intersectionID><Queue Position>

        pass
