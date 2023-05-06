import random


class Car:
    # A class variable to keep track of all cars.
    all_cars = []

    def __init__(self, id, car_queue_id, grid_size):
        Car.all_cars.append(self)
        self.id = id
        # Randomly pick a destination intersection
        self.final_destination = ""
        self.set_final_destination(grid_size)
        print("the final destination is: ", self.final_destination)
        # Set an initial balance
        self.balance = 0
        # Rush factor is random between 0 and 1
        self.rush_factor = random.random()
        self.submitted_bid = 0
        self.time_inactive = 0
        # car_queue_id is the ID of the intersection and queue the car is currently in (e.g. 11N, for intersection (1,1), north car queue).
        self.car_queue_id = car_queue_id
        self.destination_queue = self.update_destination_queue()

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
        return self.id, self.submitted_bid, self.time_inactive

    def pay_bid(self, price):
        # Price is not necessarily the same as the bid (e.g. 2nd price auction)
        self.balance -= price
        if self.balance < 0:  # This should never happen, we the bid is limited by the balance
            print("ERROR: Car {} has negative balance".format(self.id))
        print("Car {} paid {} to exit intersection".format(self.id, price))

    def set_balance(self, balance):
        # Set the balance of the car to the given balance. Used for the wage distribution.
        self.balance = balance

    def update_car_queue_id(self, new_car_queue_id):
        self.car_queue_id = new_car_queue_id

    def set_final_destination(self, grid_size):
        # Randomly pick a destination intersection
        x = random.randint(0, grid_size - 1)
        y = random.randint(0, grid_size - 1)
        self.final_destination = str(x) + str(y)

    def is_at_destination(self):
        # Evaluated whether the car is at its final destination.
        # Remove the last character from  car_queue_id (e.g. 11N -> 11).
        current_intersection = self.car_queue_id[:-1]
        if current_intersection == self.final_destination:
            return True
        else:
            return False

    def update_destination_queue(self):
        # Returns a queue ID: <intersectionID><Queue Position>
        destination_queue = ""
        current_x = int(self.car_queue_id[0])
        current_y = int(self.car_queue_id[1])
        destination_x = int(self.final_destination[0])
        destination_y = int(self.final_destination[1])

        # First, check in which direction(s) the car needs to move
        need_to_move_x = destination_x - current_x  # 0 if no need to move
        need_to_move_y = destination_y - current_y  # 0 if no need to move

        direction = ""

        if need_to_move_x and need_to_move_y:
            # If it needs to move in both directions, randomly pick one of the two
            direction = random.choice(["x", "y"])
        elif need_to_move_x:
            # If it needs to move in only one direction, pick that direction
            direction = "x"
        elif need_to_move_y:
            direction = "y"

        # Now that the direction is known, pick the appropriate queue.
        # For this, it doesn't matter from which queue the car is coming from (e.g. if it is going North, it will end up in a South queue).
        if direction == "x":
            if need_to_move_x > 0:
                destination_queue = str(current_x + 1) + str(current_y) + "E"
            else:
                destination_queue = str(current_x - 1) + str(current_y) + "W"
        elif direction == "y":
            if need_to_move_y > 0:
                destination_queue = str(current_x) + str(current_y + 1) + "N"
            else:
                destination_queue = str(current_x) + str(current_y - 1) + "S"

        print("For car {}, the destination queue is {}, with final destination {} and current location {}".format(
            self.id, destination_queue, self.final_destination, self.car_queue_id))

        self.destination_queue = destination_queue
        return self.destination_queue


def ready_for_new_epoch(self):
    self.submitted_bid = 0
