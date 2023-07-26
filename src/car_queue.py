"""This module contains the CarQueue class, which is used to represent a queue of cars at an intersection.
This class is also responsible for gathering bids from the queue and making the cars pay their individual fees.
Lastly, it also keeps track of how long the queue has been inactive (i.e. no cars have left the queue)."""


class CarQueue:
    """
    The Car Queue class is responsible for keeping track of the cars in the queue, adding/removing cars and making them pay when they win an auction.
    Attributes:
        args (argparse.Namespace): The arguments passed to the program
        id (str): The ID of the car queue, e.g. 11N for the north car queue at intersection (1,1)
        cars (list): A list of Car objects that are in the queue
        num_of_cars (int): The number of cars that are currently in the queue
        parent_intersection (Intersection): The intersection that the queue belongs to
        time_inactive (int): The number of epochs that have passed since the last car left the queue
        bids (dict): A dictionary of bids submitted by the cars in the queue. The key is the car ID,
            the value is the bid of the car
    Functions:
        get_queue_description: Returns a string describing the queue and everything in it.
        is_empty: Checks whether the queue is empty
        get_num_of_cars: Returns the number of cars in the queue
        get_time_inactive: Returns the time that has passed since the last car left the queue
        has_capacity: Checks whether the queue has capacity for more cars
        get_num_of_free_spots: Returns the number of free spots in the queue
        get_destination_of_first_car: Returns the destination of the first car in the queue
        get_parent_intersection: Returns the intersection that the queue belongs to
        add_car(car): Adds a car to the end of the queue
        remove_first_car: Removes the first car from the queue
        remove_car(car): Removes a specific car from the queue
        collect_bids(only_collect_first=True): Collects bids from all cars in the queue (not the payment, but the initial bid)
        win_auction(fee): Makes the cars in the queue pay their individual fees
        reset_bids: Resets the bids submitted by the cars in the queue
        ready_for_new_epoch: Prepares the queue for the next epoch
    """

    def __init__(self, args, parent_intersection, id):
        """ Initialize the Car queue object
        Args:
            args (argparse.Namespace): The arguments passed to the program
            parent_intersection (Intersection): The intersection that the queue belongs to
            id (str): The ID of the car queue, e.g. 11N for the north car queue at intersection (1,1)
        """
        self.args = args
        self.id = id
        self.cars = []
        self.num_of_cars = len(self.cars)
        self.parent_intersection = parent_intersection
        self.time_inactive = 0
        self.bids = {}
        self.bid_rank = 0  #  The rank of the bid compared to other queues of the intersection. Reset after every auction
        #  The rank of the time waited compared to other queues of the intersection. Reset after every auction
        self.time_waited_rank = 0

    def __str__(self):
        # Create list of all IDs of cars in the queue
        car_ids = []
        for car in self.cars:
            car_ids.append(car.id)
        return f'Car Queue (ID: {self.id}) contains cars with IDs: {car_ids}'

    def get_queue_description(self):
        """ Returns a string describing the queue and the cars in it
        Returns:
            str: A string describing the queue and the cars in it
        """
        queue_description = "Queue " + self.id + \
            " Inactivity: " + str(self.time_inactive) + "\n"
        for car in self.cars:
            queue_description += car.get_short_description() + "\n"

        return queue_description

### Helper functions ###
    def is_empty(self):
        """Checks whether the queue is empty
        Returns:
            bool: True if the queue is empty, False otherwise
        """
        return not (self.get_num_of_cars() > 0)

    def get_num_of_cars(self):
        """Returns the number of cars in the queue
        Returns:
            int: The number of cars in the queue
        """
        return len(self.cars)

    def get_time_inactive(self):
        """Returns the time that has passed since the last car left the queue
        Returns:
            int: The time that has passed since the last car left the queue
        """
        return self.time_inactive

    def has_capacity(self):
        """Returns whether the queue has capacity for more cars
        Returns:
            bool: True if the queue has capacity, False otherwise
        """
        return self.get_num_of_cars() < self.args.queue_capacity

    def get_num_of_free_spots(self):
        """Returns the number of free spots in the queue
        Returns:
            int: The number of free spots in the queue
        """
        return self.args.queue_capacity - self.get_num_of_cars()

    def get_destination_of_first_car(self):
        """Returns the destination of the first car in the queue. This is useful for the intersection to know where the car wants to go
            (e.g.to check if the new queue has capacity)
        Returns:
            str: The destination of the first car in the queue
        """
        # For now, the car is not removed. We first need to check if the new queue has capacity.
        car = self.cars[0]
        destination = car.update_next_destination_queue()
        return destination

    def get_parent_intersection(self):
        """Returns the intersection that the queue belongs to
        Returns:
            Intersection: The intersection that the queue belongs to
        """
        return self.parent_intersection

    def set_bid_rank(self, rank):
        """Sets the bid rank of the queue
        Args:
            rank (float): The rank of the queue
        """
        self.bid_rank = rank

    def get_bid_rank(self):
        """Returns the bid rank of the queue
        Returns:
            float: The bid rank of the queue
        """
        return self.bid_rank

    def set_time_waited_rank(self, rank):
        """Sets the time waited rank of the queue
        Args:
            rank (float): The rank of the queue
        """
        self.time_waited_rank = rank

    def get_time_waited_rank(self):
        """Returns the time waited rank of the queue
        Returns:
            float: The time waited rank of the queue
        """
        return self.time_waited_rank

### Queue Manipulation Functions ###
    def add_car(self, car):
        """Adds a car to the end of the queue
        Args:
            car (Car): The car to be added to the queue
        Raises:
            Exception: If the queue is full, an exception is raised
        """
        try:  # This is a sanity check
            assert self.has_capacity()
        except AssertionError:
            print("ERROR: Car Queue is full, cannot add car")
            return

        self.cars.append(car)

    def remove_first_car(self):
        """Removes and retrieves the first car from the queue
        Returns:
            Car: The first car in the queue, or None if the queue is empty
        Raises:
            Exception: If the queue is empty, an exception is raised
        """
        if self.is_empty():
            raise Exception("ERROR: Cannot remove car from empty queue")
        return self.cars.pop(0)

    def remove_car(self, car):
        """Removes a specific car from the queue
        Args:
            car (Car): The car to be removed from the queue
        Raises:
            Exception: If the car is not in the queue, an exception is raised
        """
        try:
            assert car in self.cars
        except AssertionError:
            print("ERROR: Car {} is not in queue {}".format(car.id, self.id))
            return
        self.cars.remove(car)

### Auction Functions ###
    def collect_bids(self, only_collect_first=True):
        """ Makes a collection of bids from all cars in the queue (not the payment, but the initial bid)
        Args:
            only_collect_first (bool): If True, only the first car in the queue will submit a bid.
        Returns:
            dict: A dictionary of bids submitted by the cars in the queue.
                The key is the car ID, and the value is the submitted bid of the car
        """
        # A dictionary is used so that we know which car submitted which bid
        self.bids = {}
        if only_collect_first:
            car_id, bid = self.cars[0].submit_bid()
            self.bids[car_id] = bid
        else:
            for car in self.cars:
                car_id, bid = car.submit_bid()
                self.bids[car_id] = bid

        return self.bids

    def win_auction(self, fee):
        """This is executed when the car queue has won the auction and the movement was succesful.
        Makes the cars in the queue pay their individual fees, and resets the inactivity time of the queue
        Args:
            fee (int): The total fee that the cars in the queue have to pay
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
                individual_price = fee * \
                    queue_bids[i] / total_submitted_bid

            self.cars[i].pay_bid(individual_price)
        # Finally, the inactivity time must be reset for the queue itself.
        self.time_inactive = 0

        reward = 0
        if self.parent_intersection.get_intersection_reward_type() == "time":
            reward = 1/(1+self.cars[0].get_time_at_intersection())
        elif self.parent_intersection.get_intersection_reward_type() == "time_and_urgency":
            reward = 1 / \
                (1+self.cars[0].get_time_at_intersection()
                 * self.cars[0].get_urgency())
        elif self.parent_intersection.get_intersection_reward_type() == "max_time_waited":
            reward = 1/(1+self.parent_intersection.get_max_time_waited())
        elif self.parent_intersection.get_intersection_reward_type() == "time_waited_rank":
            reward = self.get_time_waited_rank()

        self.parent_intersection.update_mechanism(reward)
        self.parent_intersection.add_reward_to_history(reward)

    def reset_bids(self):
        """Resets the bids submitted by the cars in the queue, so that the next auction can be run."""
        if self.bids != None:
            self.bids = self.bids.clear()

### Epoch Functions ###
    def ready_for_new_epoch(self):
        """Prepares the queue for the next epoch. This involved: 
            1) Resetting the bids,
            2) Updating the number of cars in the queue, 
            3) Updating the inactivity time of the queue.
            4) Reset the time waited rank of the queue and the bid rank of the queue
        """
        self.reset_bids()
        self.num_of_cars = self.get_num_of_cars()
        if not self.is_empty():
            # Inactivity time only increases if there are cars there.
            self.time_inactive += 1
        self.bid_rank = 0
        self.time_waited_rank = 0
