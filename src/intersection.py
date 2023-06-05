"""This module contains the Intersection class, which represents an intersection in the grid."""

import random

from src.car_queue import CarQueue


class Intersection:
    """
    The Intersection class is responsible for keeping track of the car queues that are part of the intersection, and for holding auctions.
    Attributes:
        id (str): The ID of the intersection, e.g. 11 for the intersection at (1,1)
        carQueues (list): A list of CarQueue objects that are part of the intersection
    Functions:
        get_intersection_description: Returns a string describing the intersection and everything in it.
        is_empty: Checks whether all car queues are empty in this intersection
        num_of_cars_in_intersection: Returns the number of cars in the intersection
        set_last_reward: Sets the last reward of the intersection
        get_last_reward: Returns the last reward of the intersection
        get_car_queue_from_intersection: Returns the car queue object given a car queue id. Car queue has to be in this intersection.
        hold_auction: Holds an auction between the car queues in this intersection. 
            Returns the id of the winning car queue and the destination (a car queue id) of the 1st car in the winning queue.
        ready_for_new_epoch: Prepares the intersection for the next epoch.
    """
    # A class variable to keep track of all intersections.
    all_intersections = []

    def __init__(self, id, queue_capacity, auction_modifier):
        """Initialize the Intersection object
        Args:
            id (str): The ID of the intersection, e.g. 11 for the intersection at (1,1)
            queue_capacity (int): The maximum number of cars that can be in a car queue
            auction_modifier (AuctionModifier): The auction modifier object that is used to modify the auction parameters
        """
        Intersection.all_intersections.append(self)
        self.id = id
        self.auction_modifier = auction_modifier
        self.carQueues = [CarQueue(queue_capacity, str(id + 'N')),
                          CarQueue(queue_capacity, str(id + 'E')),
                          CarQueue(queue_capacity, str(id + 'S')),
                          CarQueue(queue_capacity, str(id + 'W'))]
        self.last_reward = 0

    def __str__(self):
        return f'Intersection(id={self.id})'

    def get_intersection_description(self):
        """Returns a string describing the intersection and everything in it.
        Returns:
            str: A string describing the intersection and everything in it.
        """
        description = "Intersection: " + self.id + ":\n"
        for queue in self.carQueues:
            description += queue.get_queue_description()
            description += "\n"

        return description

    def is_empty(self):
        """Boolean. Checks whether all car queues are empty in this intersection
        Returns:
            bool: True if all car queues are empty, False otherwise
        """
        for queue in self.carQueues:
            if not queue.is_empty():
                return False
        return True

    def num_of_cars_in_intersection(self):
        """Returns the number of cars in the intersection
        Returns:
            int: The number of cars in the intersection
        """
        num_of_cars = 0
        for queue in self.carQueues:
            num_of_cars += queue.get_num_of_cars()
        return num_of_cars

    def set_last_reward(self, reward):
        """Sets the last reward of the intersection
        Args:
            reward (float): The reward to set
        """
        self.last_reward = reward

    def get_last_reward(self):
        """Returns the last reward of the intersection
        Returns:
            float: The last reward of the intersection
        """
        return self.last_reward

    def get_car_queue_from_intersection(self, car_queue_id):
        """Returns the car queue object given a car queue id. Car queue has to be in this intersection.
        Args:
            car_queue_id (str): The ID of the car queue (e.g. 11N)
        Returns:
            CarQueue: The car queue object with the given ID
        """
        for queue in self.carQueues:
            if queue.id == car_queue_id:
                return queue
        print("ERROR: Queue ID not found, with id: ", car_queue_id)

    def hold_auction(self, second_price=False):
        """Holds an auction between the car queues in this intersection.
        Args:
            second_price (bool): Whether to use the second price auction mechanism, instead of first-price. Defaults to False.
        Returns:
            tuple: The ID of the winning car queue and the destination (a car queue id) of the 1st car in the winning queue.
        """
        def renormalize(n, range1, range2):
            """ Normalise a value n from range1 to range2. Nested function as it is only used to normalise the bid modifier boost
            Args:
                n (float): The value to be normalised
                range1 (list): The range of the value n
                range2 (list): The range to normalise to
            Returns:
                float: The normalised value
            """
            delta1 = max(range1) - min(range1)
            if delta1 == 0:
                delta1 = 0.0001  # Avoid division by zero
            delta2 = max(range2) - min(range2)
            return (delta2 * (n - min(range1)) / delta1) + min(range2)

        collected_bids = {}
        queue_waiting_times = {}
        queue_lengths = {}
        # modification_boost_limit contains the min and max value of the final boost (e.g. max of 2 implies a boost of 2x, i.e. bid is doubled)
        queue_delay_boost, queue_length_boost, modification_boost_limit = self.auction_modifier.generate_auction_parameters()

        for queue in self.carQueues:
            if not queue.is_empty():  # Only collect bids from non-empty queues
                collected_bids[queue.id] = queue.collect_bids()
                queue_waiting_times[queue.id] = queue.get_time_inactive()
                queue_lengths[queue.id] = queue.get_num_of_cars()

        # If there is only one entry:
        if len(collected_bids) == 1:
            # We return the only queue, and its destination, and give no charge.
            winning_queue = self.get_car_queue_from_intersection(
                list(collected_bids.keys())[0])
            total_fee = 0
            destination = winning_queue.get_destination_of_first_car()
            winning_queue.set_auction_fee(total_fee)
            return winning_queue.id, destination

        # Summed_bids holds the sum of all bids for each queue
        summed_bids = {}
        for key in collected_bids.keys():
            summed_bids[key] = sum(collected_bids[key].values())

        # First calculate the initial modifications of all queues, before normalising them.
        # We need to calculate them all first, as we need the min/max value for normalisation.
        initial_modifications = {}
        for key in summed_bids.keys():  # One modified/final bid per queue
            initial_modification = queue_waiting_times[key] * \
                queue_delay_boost + queue_lengths[key] * queue_length_boost
            initial_modifications[key] = initial_modification

        # Then normalise the modifications based on the min/max values of all modifications, and the given modification_boost_limit
        final_bids = {}
        for key in summed_bids.keys():
            normalised_modification = renormalize(
                initial_modifications[key], initial_modifications.values(), modification_boost_limit)

            final_bids[key] = (1 + random.uniform(0, 0.01) +
                               summed_bids[key]) * normalised_modification

        # Winning queue is the queue with the highest bid, regardless of 1st/2nd price.
        winning_queue = self.get_car_queue_from_intersection(
            max(final_bids, key=final_bids.get))

        total_fee = 0
        if not second_price:
            # Fist price auction
            # Modifications do not count for the final fee.
            total_fee = summed_bids[winning_queue.id]

        if second_price:
            summed_bids_ordered = sorted(summed_bids.values(), reverse=True)
            total_fee = summed_bids_ordered[1]  # The 2nd highest bid

        destination = winning_queue.get_destination_of_first_car()
        winning_queue.set_auction_fee(total_fee)

        # We return the originating car queue and the destination car queue. We don't need to know the car ID,
        # as we can retrieve it later, if the move is possible.
        return winning_queue.id, destination

    def ready_for_new_epoch(self):
        """Prepares the intersection for the next epoch."""
        pass
