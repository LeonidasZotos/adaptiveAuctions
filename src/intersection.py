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
        is_empty: Checks whether all car queues are empty in this intersection
        hold_auction: Holds an auction between the car queues in this intersection. 
            Returns the id of the winning car queue and the destination (a car queue id) of the 1st car in the winning queue.
        ready_for_new_epoch: Prepares the intersection for the next epoch.
    """
    # A class variable to keep track of all intersections.
    all_intersections = []

    def __init__(self, id, queue_capacity):
        """Initialize the Intersection object
        Args:
            id (str): The ID of the intersection, e.g. 11 for the intersection at (1,1)
            queue_capacity (int): The maximum number of cars that can be in a car queue
        """
        Intersection.all_intersections.append(self)
        self.id = id
        self.carQueues = [CarQueue(queue_capacity, str(id + 'N')),
                          CarQueue(queue_capacity, str(id + 'E')),
                          CarQueue(queue_capacity, str(id + 'S')),
                          CarQueue(queue_capacity, str(id + 'W'))]

    def __str__(self):
        return f'Intersection(id={self.id})'

    def is_empty(self):
        """Boolean. Checks whether all car queues are empty in this intersection
        Returns:
            bool: True if all car queues are empty, False otherwise
        """
        for queue in self.carQueues:
            if not queue.is_empty():
                return False
        return True

    def hold_auction(self):
        """Holds an auction between the car queues in this intersection.
        Returns:
            tuple: The ID of the winning car queue and the destination (a car queue id) of the 1st car in the winning queue.
        """
        # TODO: implement auction mechanism
        # For now, we pick a random queue to be the winner
        collected_bids = {}
        for queue in self.carQueues:
            if not queue.is_empty():  # Only collect bids from non-empty queues
                collected_bids[queue.id] = queue.collect_bids()
        # Currently we don't use it, but the bids need to be set for later

        winning_queue = random.choice(
            [queue for queue in self.carQueues if not queue.is_empty()])  # Must have cars in queue
        destination = winning_queue.get_destination_of_first_car()
        total_fee = 2  # Placeholder for now
        winning_queue.set_auction_fee(total_fee)
        # We return the originating car queue and the destination car queue. We don't need to know the car ID,
        # as we can retrieve it later, if the move is possible.
        return winning_queue.id, destination

    def ready_for_new_epoch(self):
        """Prepares the intersection for the next epoch."""
        # Nothing to clear.
        pass
