"""This module contains the Car and SmallCar classes. The Car class is responsible for keeping track of the car's state, and for submitting bids.
The SmallCar class is a small version of the Car class, which only contains the essential car info. The latter is used for evaluation purposes,
so that no full copies are kept.
"""
import numpy as np
import random


class Car:
    """
    The Car class is responsible for keeping track of the car's state, and for submitting bids.
    Attributes:
        args (argparse.Namespace): Arguments parsed from the command line
        id (str): The ID of the car, e.g. 1
        car_queue_id (str): The ID of the queue the car is currently in (e.g. 11N, for intersection (1,1), North car queue).
        parent_car_queue (CarQueue): The car queue that the car is currently in.
        bidding_type (str): The type of bidding that the car uses, e.g. 'random' or 'static'.
        bid_generator (BidGenerator): The bid generator that the car uses. This is used to generate a bid.
        final_destination (str): The ID of the final destination intersection (e.g. 22, for intersection (2,2)).
        next_destination_queue (str): The ID of the queue the car is currently heading to (e.g. 22S, for intersection (2,2), South car queue).
            This is not the final destination, but the next queue the car is heading to.
        balance (float): The balance of the car. This is the amount of credit the car has left.
        urgency (float): The urgency of the car. This represents the driver's urgency.
        time_at_intersection (int): The number of epochs that the car has spent at the intersection.
        time_in_traffic_network (int): The number of epochs that the car has spent in the traffic network.
        distance_travelled_in_trip (int): The distance travelled in the current trip. Same as the number of auctions won.
        submitted_bid (float): The bid that the car submitted in the last auction.

    Functions:
        get_short_description: Returns a short description of the car, containing the ID, final destination, balance and urgency.
        is_at_destination: Checks whether the car is at its final destination. It doesn't matter in which car queue of the intersection it is.
        get_time_at_intersection: Returns the time spent at the current intersection
        get_urgency: Returns the urgency of the current car
        set_balance(new_balance): Set the balance of the car to the given balance. E.g. Used for the wage distribution.
        set_car_queue_id(new_car_queue_id): Set the queue ID of the car to a new ID E.g. Used by Grid when the car is moved.
        increase_distance_travelled_in_trip: Increase the distance spent in the current trip by 1
        calculate_satisfaction_score: Calculate the satisfaction score of the car after the completion of a trip
        reset_final_destination(): Set the final destination of the car to a new destination. E.g. Used when the car is (re)spawned.
        update_next_destination_queue: Update the next destination queue of the car. E.g. When the car participates in an auction,
            we need to know the queue where it is heading to. This function both updates the next destination queue and returns it.
        reset_car(car_queue_id): Reset the car to a new state. E.g. Used when the car is (re)spawned.
        submit_bid: Submit a bid to the auction.
        pay_bid(price): Pay the given price. Used when the car wins an auction. The price should never be higher than the balance.
            The price does not have to be the same as the submitted bid(e.g. Second-price auctions).
        ready_for_new_epoch: Prepare the car for the next epoch. This mostly clears epoch-specific variables (e.g. bids submitted)
    """

    def __init__(self, args, id, parent_car_queue, bidding_type, bid_generator):
        """ Initialize the Car object
        Args:
            args (argparse.Namespace): Arguments parsed from the command line
            id (str): The ID of the car, e.g. 1
            parent_car_queue (CarQueue): The car queue that the car is currently in.
            bidding_type (str): The type of bidding that the car uses, e.g. 'random', 'static' or RL.
            bid_generator (BidGenerator): The bidding generator that the car uses. This is used to generate a bid.
        """
        self.args = args
        self.id = id
        # Randomly pick a destination intersection
        # car_queue_id is the ID of the intersection and queue the car is currently in (e.g. 11N, for intersection (1,1), north car queue).
        self.car_queue_id = parent_car_queue.id
        self.parent_car_queue = parent_car_queue
        self.bidding_type = bidding_type
        self.bid_generator = bid_generator
        self.final_destination = ""
        self.reset_final_destination()
        # next_destination_queue is the ID of the queue the car is currently heading to (e.g. 22S, for intersection (2,2), south car queue).
        self.next_destination_queue = self.update_next_destination_queue()
        # Set an initial balance. This is set to 0 because the car will receive credit in the 0th epoch.
        self.balance = 0
        # Rush factor is random between 0 and 1, rounded to 1 decimal. The higher the urgency, the higher the urgency.
        self.urgency = self.set_urgency()
        # Time spent at the current intersection
        self.time_at_intersection = 0
        # Time spent in the network for the current trip
        self.time_in_traffic_network = 0
        # The distance travelled in the current trip. Same as the number of auctions won.
        self.distance_travelled_in_trip = 0
        # The bid that the car submitted in the last auction.
        self.submitted_bid = 0

    def __str__(self):
        return f'Car(id={self.id}), destination: {self.final_destination}, balance: {self.balance}, urgency: {self.urgency}'

    def get_short_description(self):
        """Returns a short description of the car, containing the ID, final destination, balance and urgency.
        Returns:
            str: A short description of the car, containing the ID, final destination, balance and urgency.
        """
        return f'Car(id={self.id}), destin.: {self.final_destination}, queue destin.:{self.next_destination_queue}, balance: {self.balance}, urgency: {self.urgency}, time: {self.time_in_traffic_network}'

### Helper functions ###
    def is_at_destination(self):
        """ Boolean. Checks whether the car is at its final destination. It doesn't matter in which car queue of the intersection it is.
        Returns:
            bool: True if the car is at its final destination, False otherwise
        """
        # Remove the last character from  car_queue_id (e.g. 11N -> 11), as it doesn't matter in which queue the car is.
        current_intersection = self.car_queue_id[:-1]
        if current_intersection == self.final_destination:
            return True
        else:
            return False

    def get_time_at_intersection(self):
        """ Returns the time spent at the current intersection
        Returns:
            int: The time spent at the current intersection
        """
        return self.time_at_intersection

    def get_urgency(self):
        """ Returns the urgency of the current car
        Returns:
            float: The urgency of the current car
        """
        return self.urgency

    def set_balance(self, new_balance):
        """ Set the balance of the car to the given balance. E.g. Used for the wage distribution.
        Args:
            new_balance (float): The new balance of the car.
        """
        self.balance = new_balance

    def set_car_queue_id(self, new_car_queue_id):
        """ Set the queue ID of the car to a new ID E.g. Used by Grid when the car is moved.
        Args:
            new_car_queue_id (str): The new queue ID of the car.
        """
        if self.car_queue_id != new_car_queue_id:
            self.time_at_intersection = 0
        self.car_queue_id = new_car_queue_id

    def get_parent_car_queue(self):
        """ Get the parent car queue of the car. This is the car queue that the car is currently in.
        Returns:
            CarQueue: The parent car queue of the car.
        """
        return self.parent_car_queue

    def increase_distance_travelled_in_trip(self):
        """Increase the distance spent in the current trip by 1"""
        self.distance_travelled_in_trip += 1

    def calculate_satisfaction_score(self):
        """This function should only be called at the end of a trip. Returns the satisfaction score of the trip,
            Function Explanation: The time spent in the network is divided by the distance travelled in the trip,
            to get the average time spent per intersection. This is then multiplied by the urgency.
        Returns:
            tuple: A tuple containing a small copy of the car and the satisfaction score of the trip.
                By 'small', we mean that the car only contains the necessary information.
        """
        speed = self.distance_travelled_in_trip / \
            self.time_in_traffic_network  # Distance over time is the speed
        # The higher the speed, the higher the score
        score = self.urgency * speed

        # Return a small copy of the car (only necessary information), so that the original car is not changed.
        return SmallCar(self), score

### General state functions ###
    def reset_final_destination(self):
        """Set the final destination of the car to a new destination. E.g. Used when the car is (re)spawned.
           The new destination is randomly picked and canot be the same as the current intersection.
        """

        # Randomly pick a destination intersection
        x = random.randint(0, self.args.grid_size - 1)
        y = random.randint(0, self.args.grid_size - 1)
        self.final_destination = str(x) + str(y)
        if self.car_queue_id[:-1] == self.final_destination:
            # If the car is already at its final destination, pick a new one.
            self.reset_final_destination()

    def update_next_destination_queue(self):
        """Update the next destination queue of the car. E.g. When the car participates in an auction,
           The next destination queue is picked randomly from the queues that are in the direction of the final destination.
        Returns:
            str: The ID of the queue the car is currently heading to (e.g. 22S, for intersection (2,2), south car queue).
        """
        destination_queue = ""
        # Break down the string location IDs into x and y coordinates
        current_x = int(self.car_queue_id[0])
        current_y = int(self.car_queue_id[1])
        destination_x = int(self.final_destination[0])
        destination_y = int(self.final_destination[1])

        # First, check in which direction(s) the car needs to move
        need_to_move_x = destination_x - current_x  # 0 if there is  no need to move
        need_to_move_y = destination_y - current_y  # 0 if there is no need to move

        # This holds one of the 2 possible directions, potentially picked in random if it has to move in both directions
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
        # For this, it doesn't matter from which queue the car is coming from (e.g. if it is going North, it will always end up in a South queue).
        if direction == "x":
            if need_to_move_x > 0:
                destination_queue = str(current_x + 1) + str(current_y) + "W"
            else:
                destination_queue = str(current_x - 1) + str(current_y) + "E"
        elif direction == "y":
            if need_to_move_y > 0:
                destination_queue = str(current_x) + str(current_y + 1) + "N"
            else:
                destination_queue = str(current_x) + str(current_y - 1) + "S"

        # Set the next destination queue to the newly picked queue
        self.next_destination_queue = destination_queue
        # Return the next destination queue
        return self.next_destination_queue

    def set_urgency(self):
        """Sets the urgency of the car, depending on the bidding type
        Returns: 
            urgency (float): The urgency of the car
        """
        urgency = 0
        if self.bidding_type == 'static_low':
            # Random float from gaussian with mean 0.25, and sigma 0.2
            mu_v = 0.25
            sigma_v = 0.2
            valuation = np.random.normal(mu_v, sigma_v)
            urgency = valuation
            while valuation < 0 or valuation > 1:
                valuation = np.random.normal(mu_v, sigma_v)
                urgency = valuation
        elif self.bidding_type == 'static_high':
            # Random float from gaussian with mean 0.75, and sigma 0.2
            mu_v = 0.75
            sigma_v = 0.2
            valuation = np.random.normal(mu_v, sigma_v)
            urgency = valuation
            while valuation < 0 or valuation > 1:
                valuation = np.random.normal(mu_v, sigma_v)
                urgency = valuation

        return urgency

    def reset_car(self, car_queue_id):
        """Reset the car to a new state. E.g. Used when the car is (re)spawned. This function resets the car's final destination, 
           next destination queue, urgency, submitted bid, time at intersection & time in network/trip duration. The balance is not affected.
        Args:
            car_queue_id (str): The ID of the queue the car is currently in (e.g. 11N, for intersection (1,1), north car queue).
            grid_size (int): The size of the grid (e.g. 3 for a 3x3 grid). This is used to pick a new valid final destination.
        """
        self.car_queue_id = car_queue_id
        self.reset_final_destination()
        self.next_destination_queue = self.update_next_destination_queue()
        self.urgency = self.set_urgency()
        self.submitted_bid = 0
        self.time_at_intersection = 0
        self.time_in_traffic_network = 0
        self.distance_travelled_in_trip = 0

### Auction functions ###
    def submit_bid(self):
        """Submit a bid to the auction.
        Returns:
            self.id (str): The ID of the car, e.g. 1. This is included so that the intersection can keep track of which car submitted which bid.
            self.submitted_bid (float): The bid that the car submits in the auction.
        Raises:
            Exception: If the car tries to submit a negative bid, an exception is raised.
        """
        self.submitted_bid = self.bid_generator.generate_bid(
            self.bidding_type, self.balance, self.urgency)
        # If there is not enough balance, bid entire balance.
        if self.submitted_bid > self.balance:
            self.submitted_bid = self.balance
        if self.submitted_bid < 0:
            raise Exception("ERROR: Car {} tried to submit a negative bid {}".format(
                self.id, self.submitted_bid))
        # Return the car's id and the bid
        return self.id, self.submitted_bid

    def pay_bid(self, price):
        """Pay the given price. Used when the car wins an auction. The price should never be higher than the balance.
           The price does not have to be the same as the submitted bid(e.g. Second-price auctions).
        Args:
            price (float): The price that the car has to pay (i.e. amount to deduct from balance).
        Raises:
            Exception: If the price is higher than the balance, an exception is raised.
            Exception: If the balance is negative, an exception is raised.
        """
        try:
            assert price <= self.balance
        except AssertionError:
            print("ERROR: Car {} had to pay more than its balance (price: {}, balance: {}, bidded: {})".format(
                self.id, price, self.balance, self.submitted_bid))

        self.balance -= price
        try:
            assert self.balance >= 0
        except AssertionError:
            print("ERROR: Car {} has negative balance (balance: {}, price: {}, bidded: {})".format(
                self.id, self.balance, price, self.submitted_bid))

    def ready_for_new_epoch(self):
        """Prepare the car for the next epoch. This mostly clears epoch-specific variables (e.g. bids submitted)"""
        self.time_at_intersection += 1
        self.time_in_traffic_network += 1
        self.submitted_bid = 0


class SmallCar:
    """
    The SmallCar class is a small version of the Car class, which only contains the essential car info. This class is used for evaluation purposes, 
    so that no full copies of cars are kept.
    Attributes:
        bidding_type (str): The type of bidding that the car uses, e.g. 'random' or 'static'.
        bid_generator (BidGenerator): The bid generator that the car uses. This is used to generate a bid.
        balance (float): The balance of the car. This is the amount of credit the car has left.
        urgency (float): The urgency of the car. This represents the driver's urgency.
        time_at_intersection (int): The number of epochs that the car has spent at the intersection.
        time_in_traffic_network (int): The number of epochs that the car has spent in the traffic network.
        distance_travelled_in_trip (int): The distance travelled in the current trip. Same as the number of auctions won.
        submitted_bid (float): The bid that the car submitted in the last auction.
    """

    def __init__(self, Car):
        """ Initialize the SmallCar object
        Args:
            Car (Car): The car object to copy.
        """
        self.bidding_type = Car.bidding_type
        self.bid_generator = Car.bid_generator
        self.balance = Car.balance
        self.urgency = Car.urgency
        self.time_at_intersection = Car.time_at_intersection
        self.time_in_traffic_network = Car.time_in_traffic_network
        self.distance_travelled_in_trip = Car.distance_travelled_in_trip
        self.submitted_bid = Car.submitted_bid
