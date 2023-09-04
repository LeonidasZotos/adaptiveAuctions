"""This module contains the Intersection class, which represents an intersection in the grid."""

import random
import numpy as np
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
        get_num_of_non_empty_queues: Returns the number of non-empty queues in the intersection
        get_max_time_waited: Returns the maximum time waited by a car_queue in the intersection
        get_last_reward: Returns the last reward of the intersection
        add_reward_to_history: Adds a reward to the reward history
        add_max_time_waited_to_history: Adds the maximum time waited by any car_queue in the intersection to the history
        get_auction_reward_history: Returns the reward history of the auction modifier
        get_auction_throughput_history: Returns the throughput history of the auction modifier
        get_max_time_waited_history: Returns the maximum time waited history of the auction modifier
        get_auction_parameters_and_valuations_and_counts: Returns the parameters, their valuations and counts of the auction
        calc_inact_rank: Calculates the inactivity rank of each queue, and sets it.
        calc_bid_rank: Calculates the bid rank of each queue, and sets it.
        hold_auction(): Holds an auction between the car queues in this intersection. 
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
        self.last_tried_auction_params = [nan]
        self.reward_history = []
        # Each element is either 0 or 1, depending on whether a car passed through the intersection in that epoch
        self.throughput_history = []
        self.max_time_waited_history = []
        self.gini_history = []
        # A list of inact and bid ranks of winners
        self.winners_bid_ranks = []
        self.winners_inact_ranks = []

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

    def get_last_tried_auction_params(self):
        """Returns the last tried auction parameters
        Returns:
            list: The last tried auction parameters
        """
        return self.last_tried_auction_params

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

    def get_num_of_non_empty_queues(self):
        """Returns the number of non-empty queues in the intersection
        Returns:
            int: The number of non-empty queues in the intersection
        """
        return sum([not queue.is_empty() for queue in self.carQueues])

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

    def add_gini_to_history(self):
        """Adds the GINI coefficient to the history"""
        def calc_gini(x):
            """Calculates the GINI coeffecient of a list of numbers.
            Source: https://www.statology.org/gini-coefficient-python/
            """
            x = np.array(x)
            total = 0
            for i, xi in enumerate(x[:-1], 1):
                total += np.sum(np.abs(xi - x[i:]))
                with np.errstate(divide='ignore', invalid='ignore'):
                    gini_result = total / (len(x)**2 * np.mean(x))
            return gini_result
        # First, calculate the gini coefficient from the times waited.
        gini_value = nan
        if self.get_num_of_non_empty_queues() > 1:
            times_waited = [queue.get_time_inactive()
                            for queue in self.carQueues]
            gini_value = calc_gini(times_waited)

        self.gini_history.append(gini_value)

    def add_winner_bid_rank(self, bid_rank):
        """Adds the bid rank of the winner to the history"""
        self.winners_bid_ranks.append(bid_rank)

    def add_winner_inact_rank(self, inact_rank):
        """Adds the inactivity rank of the winner to the history"""
        self.winners_inact_ranks.append(inact_rank)

    def calc_and_get_mean_winners_bid_ranks(self):
        """Returns the bid ranks of the winners
        Returns:
            list: The bid ranks of the winners
        """
        if len(self.winners_bid_ranks) == 0:
            return nan
        mean = sum(self.winners_bid_ranks) / len(self.winners_bid_ranks)
        return mean

    def calc_and_get_mean_winners_inact_ranks(self):
        """Returns the inactivity ranks of the winners
        Returns:
            list: The inactivity ranks of the winners
        """
        if len(self.winners_inact_ranks) == 0:
            return nan
        mean = sum(self.winners_inact_ranks) / len(self.winners_inact_ranks)
        return mean

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

    def get_gini_history(self):
        """Returns the GINI coefficient history of the intersection
        Returns:
            list: The GINI coefficient history of the intersection
        """
        return self.gini_history

    def get_auction_parameters_and_valuations_and_counts(self):
        """Returns the parameters, their valuations and counts of the auction
            Returns:
                tuple: A tuple containing the parameters, their valuations and counts of the auction  
            """
        return self.auction_modifier.get_parameters_and_valuations_and_counts()

    def calc_inact_rank(self):
        """This function calculates the inactivity rank of each queue, and sets it.
        The rank is relative to the inactivity time of the other queues. The higher the inactivity, the higher the rank. This leads to a larger bid boost.
        """
        ordered_queues = []
        for queue in self.carQueues:
            if not queue.is_empty():
                ordered_queues.append(queue)

        num_of_queues = len(ordered_queues)
        ordered_queues = sorted(
            ordered_queues, key=lambda queue: queue.get_time_inactive())
        for index, queue in enumerate(ordered_queues):
            queue.set_inact_rank(index / num_of_queues)

        # If they are equal, give them the same rank.
        for index, queue in enumerate(ordered_queues):
            if index != 0 and queue.get_time_inactive() == ordered_queues[index-1].get_time_inactive():
                queue.set_inact_rank(
                    ordered_queues[index-1].get_inact_rank())
        # Normalise between 0 and 1
        max_inact_rank = max(
            [queue.get_inact_rank() for queue in ordered_queues])
        min_inact_rank = min(
            [queue.get_inact_rank() for queue in ordered_queues])

        if max_inact_rank == min_inact_rank:
            # If all times are equal, we set all ranks to 0.5
            for queue in ordered_queues:
                queue.set_inact_rank(0.5)
        else:
            for queue in ordered_queues:
                queue.set_inact_rank(
                    (queue.get_inact_rank() - min_inact_rank) / (max_inact_rank - min_inact_rank))

    def calc_bid_rank(self, summed_bids):
        """This function calculates the bid rank of each queue, and sets it. The rank is relative to the bid of the other queues,
        higher->better->bigger boost. 
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

        # Normalise them between 0 and 1
        max_bid_rank = max([queue.get_bid_rank() for queue in ordered_queues])
        min_bid_rank = min([queue.get_bid_rank() for queue in ordered_queues])
        if max_bid_rank == min_bid_rank:
            # If all bids are equal, we set all ranks to 0.5
            for queue in ordered_queues:
                queue.set_bid_rank(0.5)
        else:
            for queue in ordered_queues:
                queue.set_bid_rank(
                    (queue.get_bid_rank() - min_bid_rank) / (max_bid_rank - min_bid_rank))

    def hold_auction(self):
        """Holds an auction between the car queues in this intersection. Modifies self.auction_fees. 
        Returns:
            tuple: A tuple containing two arrays. The first array contains the IDs of the winning queues, and the second 
                array contains the destinations of the first car in each queue.
        """
        collected_bids = {}

        for queue in self.carQueues:
            # Only collect bids from non-empty queues.
            if not queue.is_empty():
                # Only collect the first car's bid.
                collected_bids[queue.id] = queue.collect_bids(
                    only_collect_first=True)

        # If there is only one populated queue, no auction needs to be held.
        if len(collected_bids) == 1:
            # We return the only queue, and its destination, and do not charge a fee.
            winning_queue = utils.get_car_queue_from_intersection(
                self, list(collected_bids.keys())[0])
            destination = winning_queue.get_destination_of_first_car()
            self.auction_fees = [0]
            return [winning_queue.id], [destination]

        # An actual auction is held:
        queue_delay_boost = self.auction_modifier.select_auction_params()
        self.last_tried_auction_params = [queue_delay_boost]

        # Summed_bids holds the sum of all bids for each queue
        summed_bids = {}
        for key in collected_bids.keys():
            summed_bids[key] = sum(collected_bids[key].values())
        # Calculate the inactivity and bid ranks, which will later be used for the reward
        self.calc_inact_rank()
        self.calc_bid_rank(summed_bids)

        final_bids = {}
        # One modified/final bid per queue. Small noise is added to avoid ties.
        for key in summed_bids.keys():
            queue = utils.get_car_queue_from_intersection(self, key)
            if queue_delay_boost != 1:
                final_bids[key] = (summed_bids[key] + (queue.get_inact_rank() /
                                                       (1 - queue_delay_boost))) + random.uniform(0, 0.00001)
            else:  # Avoid division by 0
                final_bids[key] = (summed_bids[key] + (queue.get_inact_rank() /
                                                       (1 - 0.9999))) + random.uniform(0, 0.00001)

        # Winning queue is the queue with the highest bid. They pay the 2nd highest bid/price.
        # Order the bids in descending order. Since python 3.7 dictionaries are ordered.
        final_bids = {k: v for k, v in sorted(
            final_bids.items(), key=lambda item: item[1], reverse=True)}

        queues_in_order_of_final_bids = [utils.get_car_queue_from_intersection(
            self, queue_id) for queue_id in final_bids.keys()]

        queues_in_order_of_original_bids = [utils.get_car_queue_from_intersection(
            self, queue_id) for queue_id in summed_bids.keys()]

        # All fees that are higher than the winners fee, are set to 0, so that the winner doesn't have to pay more than they bid.
        # For example, with 3 bidders, if the winner bid the lowest, they shouldn't pay the 2nd price, but instead 0 (2nd highest price below the winner's bid)
        for queue in queues_in_order_of_original_bids:
            if summed_bids[queue.id] > summed_bids[queues_in_order_of_final_bids[0].id]:
                summed_bids[queue.id] = 0

        # order depending on the summed bid
        queues_in_order_of_original_bids = sorted(
            queues_in_order_of_original_bids, key=lambda queue: summed_bids[queue.id], reverse=True)

        # All fees that are higher than the winners fee, are set to 0, so that the winner doesn't have to pay more than they bid.
        # For example, with 3 bidders, if the winner bid the lowest, they shouldn't pay the 2nd price, but instead 0 (2nd highest price below the winner's bid)
        for queue in queues_in_order_of_original_bids:
            if summed_bids[queue.id] > summed_bids[queues_in_order_of_final_bids[0].id]:
                summed_bids[queue.id] = 0

        self.auction_fees = [summed_bids[queue.id]
                             for queue in queues_in_order_of_original_bids]
        # Remove the highest bid as we have a 2nd price auction
        self.remove_top_fee()

        destinations = [queue.get_destination_of_first_car()
                        for queue in queues_in_order_of_final_bids]
        winning_queues_id = [
            queue.id for queue in queues_in_order_of_final_bids]

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
        self.auction_modifier.update_expected_rewards(
            self.last_tried_auction_params, reward)

    def ready_for_new_epoch(self, epoch):
        """Prepares the intersection for the next epoch.
        Args:
            epoch (int): The current epoch
        """
        self.auction_fees = []
        self.last_tried_auction_params = [nan]
        if len(self.reward_history) <= epoch:
            self.reward_history.append(nan)
        if len(self.throughput_history) <= epoch:
            self.throughput_history.append(0)
        self.add_max_time_waited_to_history()
        self.add_gini_to_history()
