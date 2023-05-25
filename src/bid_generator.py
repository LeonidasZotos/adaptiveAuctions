"""This module contains the BidGenerator class, which contains the modofier that is used to modify the bidding behaviour of the cars"""
import random


class BidGenerator:
    """
    This is the BidGenerator class. The role of the generator is to generate bids for the cars, based on the bidding strategy.
    Attributes:

    Functions:

    """
    static_bid = 10  # To be adjusted manually.

    def __init__(self,):
        """Initialize the BidGenerator object
        Args:

        """

    def __str__(self):
        return f'Bid Generator'

    def generate_static_bid(self, rush_factor, balance):
        """Returns a static bid, multiplied by the rush factor
        Args:
            rush_factor (float): The rush factor of the car.
            balance (float): The balance of the car.
        """
        return self.static_bid * rush_factor

    def generate_random_bid(self, balance):
        """Returns a random bid between 0 and the total balance of the car.
        Args:
            balance (float): The balance of the car.
        """
        return random.uniform(0, balance)

    def generate_RL_bid(self, balance, rush_factor):
        """TODO: Returns a bid, based on the RL bidding strategy. For now, return static bid
        Args:
            balance (float): The balance of the car.
        """
        return self.generate_static_bid(rush_factor, balance)

    def generate_bid(self, bidding_strategy, balance, rush_factor):
        """Returns a bid, based on the bidding strategy.
        Args:
            bidding_strategy (string): The bidding strategy of the car.
            balance (float): The balance of the car.
            rush_factor (float): The rush factor of the car.
        """

        if bidding_strategy == 'static':
            return self.generate_static_bid(rush_factor, balance)
        if bidding_strategy == 'random':
            return self.generate_random_bid(balance)
        if bidding_strategy == 'RL':
            return self.generate_RL_bid(balance, rush_factor)
        else:
            print("ERROR: Invalid bidding strategy: ", bidding_strategy,  ". Returning random bid.")
            return self.generate_random_bid(balance)

    def ready_for_new_epoch(self):
        """Prepares the Bid Generator for the next epoch."""
        # Nothing to update.
        pass
