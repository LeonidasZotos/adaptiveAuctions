"""This module contains the CarQueue class, which is used to represent a queue of cars at an intersection.
This class is also responsible for gathering bids from the queue and making the cars pay their individual fees.
Lastly, it also keeps track of how long the queue has been inactive (i.e. no cars have left the queue)."""


class CarQueue:
    """
    The Car Queue class is responsible for keeping track of the cars in the queue, adding/removing cars and making them pay when they win an auction.
    Attributes:
        id (str): The ID of the car queue, e.g. 11N for the north car queue at intersection (1,1)
        cars (list): A list of Car objects that are in the queue
        capacity (int): The maximum number of cars that can be in the queue
        num_of_cars (int): The number of cars that are currently in the queue
        time_inactive (int): The number of epochs that have passed since the last car left the queue
        bids (dict): A dictionary of bids submitted by the cars in the queue. The key is the car ID,
            the value is the bid of the car
        total_fee (int): The total fee that the cars in the queue have to pay, in case they win an auction
    Functions:
        is_empty: Checks whether the queue is empty
        get_num_of_cars: Returns the number of cars in the queue
        get_time_inactive: Returns the time that has passed since the last car left the queue
        has_capacity: Checks whether the queue has capacity for more cars
        get_num_of_free_spots: Returns the number of free spots in the queue
        get_destination_of_first_car: Returns the destination of the first car in the queue
        set_auction_fee: Sets the total fee that the cars in the queue have to pay, in case they win an auction
        add_car: Adds a car to the end of the queue
        remove_first_car: Removes the first car from the queue
        remove_car: Removes a specific car from the queue
        collect_bids: Collects bids from all cars in the queue (not the payment, but the initial bid)
        win_auction: Makes the cars in the queue pay their individual fees
        reset_bids: Resets the bids submitted by the cars in the queue
        ready_for_new_epoch: Prepares the queue for the next epoch
    """

    # A class variable to keep track of all car queues.
    all_car_queues = []

    def __init__(self, max_capacity, id):
        """ Initialize the Car queue object
        Args:
            max_capacity (int): The maximum number of cars that can be in the queue
            id (str): The ID of the car queue, e.g. 11N for the north car queue at intersection (1,1)
        """
        CarQueue.all_car_queues.append(self)
        self.id = id
        self.cars = []
        self.capacity = max_capacity
        self.num_of_cars = len(self.cars)
        self.bids = {}
        self.total_fee = 0
        self.time_inactive = 0

    def __str__(self):
        # Create list of all IDs of cars in the queue
        car_ids = []
        for car in self.cars:
            car_ids.append(car.id)

        return f'Car Queue (ID: {self.id}) contains cars with IDs: {car_ids}'

### Helper functions ###
    def is_empty(self):
        """Checks whether the queue is empty
        Returns:
            bool: True if the queue is empty, False otherwise
        """
        return not (self.get_num_of_cars() > 0)

    def get_num_of_cars(self):
        """Returns the number of cars in the queue"""
        return len(self.cars)

    def get_time_inactive(self):
        """Returns the time that has passed since the last car left the queue"""
        return self.time_inactive

    def has_capacity(self):
        """Returns whether the queue has capacity for more cars"""
        return self.get_num_of_cars() < self.capacity

    def get_num_of_free_spots(self):
        """Returns the number of free spots in the queue"""
        return self.capacity - self.get_num_of_cars()

    def get_destination_of_first_car(self):
        """Returns the destination of the first car in the queue. This is useful for the intersection to know where the car wants to go
            (e.g.to check if the new queue has capacity)
        """
        # For now, the car is not removed. We first need to check if the new queue has capacity.
        car = self.cars[0]
        destination = car.update_next_destination_queue()
        return destination

    def set_auction_fee(self, fee):
        """Sets the total fee that the cars in the queue have to pay in total, in case they win an auction
        Args:
            fee (int): The total fee that the cars in the queue have to pay
        """
        self.total_fee = fee

### Queue Manipulation Functions ###
    def add_car(self, car):
        """Adds a car to the end of the queue
            Args:
                car (Car): The car to be added to the queue
        """
        # The queue should not be full when this function is called, but this is a sanity check
        if self.has_capacity():
            self.cars.append(car)
        else:
            print("ERROR: Car Queue is full, cannot add car")

    def remove_first_car(self):
        """Removes and retrieves the first car from the queue
            Returns:
                Car: The first car in the queue, or None if the queue is empty
        """
        if self.is_empty():
            print("ERROR: Cannot remove car from empty queue")
            return None
        return self.cars.pop(0)

    def remove_car(self, car):
        """Removes a specific car from the queue
            Args:
                car (Car): The car to be removed from the queue
        """
        if car in self.cars:
            self.cars.remove(car)
        else:
            print("ERROR: Car {} is not in queue {}".format(car.id, self.id))

### Auction Functions ###
    def collect_bids(self):
        """ Makes a collection of bids from all cars in the queue (not the payment, but the initial bid)
        Returns:
            dict: A dictionary of bids submitted by the cars in the queue.
                The key is the car ID, and the value is the submitted bid of the car
        """
        self.bids = {}
        # A dictionary is used so that we know which car submitted which bid
        for car in self.cars:
            car_id, bid = car.submit_bid()
            self.bids[car_id] = bid
        return self.bids

    def win_auction(self):
        """This is executed when the car queue has won the auction and the movement was succesful.
        Makes the cars in the queue pay their individual fees, and resets the inactivity time of the queue
        """
        # If a queue wins an auction:
        # 1. The winning bid is paid by the cars in the queue.
        # 2. The inactivity time is reset for the queue.

        # First, the bid must be paid
        queue_car_ids = []  # This holds the IDs of all cars in the queue
        queue_bids = []  # This holds the bids of all cars in the queue, in the same order as the IDs
        total_submitted_bid = 0  # This is the sum of the bids of all cars in the queue

        # First, separate the bids and the car IDs into two lists, from the bids that were previously collected.
        for bid in self.bids.items():
            queue_car_ids.append(bid[0])
            queue_bids.append(bid[1])
            total_submitted_bid += bid[1]

        # Second, pay the bids for all cars in the queue. The payment is proportional to the individual bid.
        for i in range(len(queue_car_ids)):
            # The winning bid is divided proportionally depending on the individual bids of the cars in the queue.
            # Default case is that the car pays nothing (This is explicit to avoid division by zero)
            individual_price = 0
            if total_submitted_bid > 0:
                individual_price = self.total_fee * \
                    queue_bids[i] / total_submitted_bid

            self.cars[i].pay_bid(individual_price)
        # Finally, the inactivity time must be reset for the queue itself.
        self.time_inactive = 0

    def reset_bids(self):
        """Resets the bids submitted by the cars in the queue, so that the next auction can be run."""
        if self.bids != None:
            self.bids = self.bids.clear()

    ### Epoch Functions ###
    def ready_for_new_epoch(self):
        """Prepares the queue for the next epoch. This involved: 
            1) Resetting the bids,
            2) Updating the number of cars in the queue, 
            3) Updating the inactivity time of the queue."""
        self.reset_bids()
        self.num_of_cars = self.get_num_of_cars()
        self.time_inactive += 1
        self.total_fee = 0
