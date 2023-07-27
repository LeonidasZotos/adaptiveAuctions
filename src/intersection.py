"""This module contains the Intersection class, which represents an intersection in the grid."""

import random
from math import nan, isnan

import src.utils as utils
from src.car_queue import CarQueue


class Intersection:
    """
    The Intersection class is responsible for keeping track of the car queues that are part of the intersection, and for holding auctions.
    Attributes:
        args (argparse.Namespace): Arguments parsed from the command line
        id (str): The ID of the intersection, e.g. 11 for the intersection at (1,1)
        auction_modifier (AuctionModifier): The auction modifier object that is used to modify the auction parameters
        carQueues (list): A list of CarQueue objects that are part of the intersection
        auction_fees (list): A list of fees collected from the auctions held in this intersection, ordered in descending order
        last_tried_auction_params (list): A list of the last tried auction parameters
        reward_history (list): A list of rewards collected by the intersection
        throughput_history (list): A list of throughput values collected by the intersection
        max_time_waited_history (list): A list of the maximum_time_waited values collected by the intersection for all epochs.
    Functions:
        get_intersection_description: Returns a string describing the intersection and everything in it.
        get_car_queues: Returns the car queues that are part of this intersection
        get_auction_fee: Returns the fee that should be paid.
        get_intersection_reward_type: Returns the type of reward for the intersection.
        remove_top_fee: Removes the top/highest fee from the list of fees.
        is_empty: Checks whether all car queues are empty in this intersection
        get_num_of_cars_in_intersection: Returns the number of cars in the intersection
        get_max_time_waited: Returns the maximum time waited by a car_queue in the intersection
        get_last_reward: Returns the last reward of the intersection
        add_reward_to_history: Adds a reward to the reward history
        add_max_time_waited_to_history: Adds the maximum time waited by any car_queue in the intersection to the history
        get_auction_reward_history: Returns the reward history of the auction modifier
        get_auction_throughput_history: Returns the throughput history of the auction modifier
        get_max_time_waited_history: Returns the maximum time waited history of the auction modifier
        get_auction_parameters_and_valuations: Returns the parameters and their valuations of the auction
        calc_inact_rank: Calculates the inactivity rank of each queue, and sets it.
        calc_bid_rank: Calculates the bid rank of each queue, and sets it.
        hold_auction(second_price=False): Holds an auction between the car queues in this intersection. 
            Returns the id of the winning car queue and the destination (a car queue id) of the 1st car in the winning queue.
        update_mechanism(reward): Updates the auction modifier mechanism
        ready_for_new_epoch: Prepares the intersection for the next epoch.
    """

    def __init__(self, args, id, auction_modifier):
        """Initialize the Intersection object
        Args:
            args (argparse.Namespace): Arguments parsed from the command line
            id (str): The ID of the intersection, e.g. 11 for the intersection at (1,1)
            auction_modifier (AuctionModifier): The auction modifier object that is used to modify the auction parameters
        """
        self.args = args
        self.id = id
        self.auction_modifier = auction_modifier
        self.carQueues = [CarQueue(args, self, str(id + 'N')),
                          CarQueue(args, self, str(id + 'E')),
                          CarQueue(args, self, str(id + 'S')),
                          CarQueue(args, self, str(id + 'W'))]
        self.auction_fees = []
        self.last_tried_auction_params = [-1, -1, -1]
        self.reward_history = []
        # Each element is either 0 or 1, degpending on whether a car passed through the intersection in that epoch
        self.throughput_history = []
        self.max_time_waited_history = []

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

    def get_car_queues(self):
        """Returns the car queues that are part of this intersection
        Returns:
            list: A list of CarQueue objects that are part of this intersection
        """
        return self.carQueues

    def get_auction_fee(self):
        """Returns the fee that should be paid.
        Returns:
            float: The fee that should be paid
        """
        if len(self.auction_fees) == 0:  # This can happen if we use 2nd price and we have removed fees due to not-possible movements.
            return 0
        return self.auction_fees[0]

    def get_intersection_reward_type(self):
        """Returns the type of reward for the intersection. 
        Returns:
            str: The type of reward for the intersection.
        """
        return self.args.intersection_reward_type

    def remove_top_fee(self):
        """Removes the top/highest fee from the list of fees."""
        if len(self.auction_fees) > 0:
            self.auction_fees.pop(0)
        else:  # If There are no fees to remove, we set the fee to 0
            self.auction_fees = [0]

    def is_empty(self):
        """Boolean. Checks whether all car queues are empty in this intersection
        Returns:
            bool: True if all car queues are empty, False otherwise
        """
        for queue in self.carQueues:
            if not queue.is_empty():
                return False
        return True

    def get_num_of_cars_in_intersection(self):
        """Returns the number of cars in the intersection
        Returns:
            int: The number of cars in the intersection
        """
        return sum([queue.get_num_of_cars() for queue in self.carQueues])

    def get_max_time_waited(self):
        """Returns the maximum time waited by a car_queue in the intersection
        Returns:
            float: The maximum time waited by a car_queue in the intersection
            or nan  if the intersection is empty
        """
        if self.is_empty():
            return nan

        all_times = [queue.get_time_inactive() for queue in self.carQueues]

        return max(all_times)

    def get_last_reward(self):
        """Returns the last reward of the intersection
        Returns:
            float: The last reward of the intersection
        """
        if len(self.reward_history) == 0:
            return 0

        # Find the last non-nan value. The list is filled with nans for the epochs where there was no reward/no movement happened
        last_non_nan = 0
        for value in reversed(self.reward_history):
            if not isinstance(value, float) or not isnan(value):
                last_non_nan = value
                break
        return last_non_nan

    def add_reward_to_history(self, reward):
        """Adds a reward to the reward history"""
        self.reward_history.append(reward)

    def add_max_time_waited_to_history(self):
        """Adds the maximum time waited by any car_queue in the intersection to the history"""
        self.max_time_waited_history.append(self.get_max_time_waited())

    def get_auction_reward_history(self):
        """Returns the reward history of the auction modifier
        Returns:
            list: The reward history of the auction modifier
        """
        return self.reward_history

    def get_auction_throughput_history(self):
        """Returns the throughput history of the intersection
        Returns:
            list: The throughput history of the intersection
        """
        return self.throughput_history

    def get_max_time_waited_history(self):
        """Returns the maximum time waited history of the intersection
        Returns:
            list: The maximum time waited history of the intersection
        """
        return self.max_time_waited_history

    def get_auction_parameters_and_valuations(self):
        """Returns the parameters and their valuations of the auction
            Returns:
                None or
                tuple: A tuple containing the parameters and their valuations of the auction  
            """
        return self.auction_modifier.get_parameters_and_valuations()

    def calc_inact_rank(self):
        """This function calculates the inactivity rank of each queue, and sets it.
        The rank is relative to the inactivity time of the other queues. The higher the inactivity, the higher the rank. This leads to a larger bid boost.
        """
        ordered_queues = []
        for queue in self.carQueues:
            # Only collect bids from non-empty queues.
            if not queue.is_empty():
                ordered_queues.append(queue)

        num_of_queues = len(ordered_queues)
        ordered_queues = sorted(
            ordered_queues, key=lambda queue: queue.get_time_inactive())
        for index, queue in enumerate(ordered_queues):
            queue.set_time_waited_rank(index / num_of_queues)

        # If they are equal, give them the same rank.
        for index, queue in enumerate(ordered_queues):
            if index != 0 and queue.get_time_inactive() == ordered_queues[index-1].get_time_inactive():
                queue.set_time_waited_rank(
                    ordered_queues[index-1].get_time_waited_rank())

    def calc_bid_rank(self, summed_bids):
        """This function calculates the bid rank of each queue, and sets it. The rank is relative to the bid of the other queues, higher->better->bigger boost
        """
        queue_ids = list(summed_bids.keys())
        num_of_queues = len(queue_ids)
        # Order the qeueus by their bids
        ordered_queues = [utils.get_car_queue_from_intersection(
            self, queue_id) for queue_id in queue_ids]
        ordered_queues = sorted(
            ordered_queues, key=lambda queue: summed_bids[queue.id], reverse=False)
        for index, queue in enumerate(ordered_queues):
            queue.set_bid_rank(index / num_of_queues)

        # If they are equal, give them the same rank.
        for index, queue in enumerate(ordered_queues):
            if index != 0 and summed_bids[queue.id] == summed_bids[ordered_queues[index-1].id]:
                queue.set_bid_rank(
                    ordered_queues[index-1].get_bid_rank())

    def hold_auction(self):
        """Holds an auction between the car queues in this intersection. Modifies self.auction_fees. 
        Returns:
            tuple: A tuple containing two arrays. The first array contains the IDs of the winning queues, and the second 
                array contains the destinations of the first car in each queue.
        """
        collected_bids = {}
        queue_waiting_times = {}
        queue_lengths = {}
        queue_delay_boost = self.auction_modifier.generate_auction_parameters()
        self.last_tried_auction_params = [queue_delay_boost]

        for queue in self.carQueues:
            # Only collect bids from non-empty queues.
            if not queue.is_empty():
                # Only collect the first car's bid.
                collected_bids[queue.id] = queue.collect_bids(
                    only_collect_first=True)
                queue_waiting_times[queue.id] = queue.get_time_inactive()
                queue_lengths[queue.id] = queue.get_num_of_cars()
        # If there is only one entry:
        if len(collected_bids) == 1:
            # We return the only queue, and its destination, and do not charge a fee.
            winning_queue = utils.get_car_queue_from_intersection(
                self, list(collected_bids.keys())[0])
            destination = winning_queue.get_destination_of_first_car()
            self.auction_fees = [0]
            return [winning_queue.id], [destination]

        # Summed_bids holds the sum of all bids for each queue
        summed_bids = {}
        for key in collected_bids.keys():
            summed_bids[key] = sum(collected_bids[key].values())

        # Calculate the inactivity and bid ranks, which will later be used for the reward
        self.calc_inact_rank()
        self.calc_bid_rank(summed_bids)
        
        # Calculate the final boosted bids
        final_bids = {}
        # One modified/final bid per queue. Small noise is added to avoid ties.
        for key in summed_bids.keys():
            queue = utils.get_car_queue_from_intersection(self, key)
            final_bids[key] = (summed_bids[key] + (queue.get_time_waited_rank() * queue_delay_boost)) + random.uniform(0, 0.001)

        # Winning queue is the queue with the highest bid. They pay the 2nd highest bid/price.
        # Order the bids in descending order. Since python 3.7 dictionaries are ordered.
        final_bids = {k: v for k, v in sorted(
            final_bids.items(), key=lambda item: item[1], reverse=True)}

        queues_in_order = [utils.get_car_queue_from_intersection(
            self, queue_id) for queue_id in final_bids.keys()]

        self.auction_fees = [summed_bids[queue.id]
                             for queue in queues_in_order]


        # Remove the highest bid as we have a 2nd price auction
        self.remove_top_fee()

        destinations = [queue.get_destination_of_first_car()
                        for queue in queues_in_order]
        winning_queues_id = [queue.id for queue in queues_in_order]

        # We return the originating car queues and the destination car queues. We don't need to know the car ID,
        # as we can retrieve it later, if the move is possible.
        if self.args.only_winning_bid_moves:
            # If we only care about the winning bid, instead of all bids, we only return the winning bid
            # This doesn't consider other movements in case the winning bidder's movement is not possible.
            destinations = destinations[:1]
            winning_queues_id = winning_queues_id[:1]

        return winning_queues_id, destinations

    def update_mechanism(self, reward):
        """Updates the auction modifier mechanism
        Args:
            reward (float): The reward of the last auction
        """
        if self.args.auction_modifier_type == "bandit":
            self.auction_modifier.update_bandit_params(
                self.last_tried_auction_params, reward)

    def ready_for_new_epoch(self, epoch):
        """Prepares the intersection for the next epoch.
        Args:
            epoch (int): The current epoch
        """
        self.auction_fees = []
        if len(self.reward_history) <= epoch:
            self.reward_history.append(nan)
        if len(self.throughput_history) <= epoch:
            self.throughput_history.append(0)
        self.add_max_time_waited_to_history()
