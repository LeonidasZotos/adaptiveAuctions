"""This module contains the BidGenerator class, which contains the modofier that is used to modify the bidding behaviour of the cars"""
import random


class BidGenerator:
    """
    This is the BidGenerator class. The role of the generator is to generate bids for the cars, based on the bidding strategy.
    Attributes:
        -
    Functions:
        generate_static_bid(rush_factor): Returns a static bid, multiplied by the rush factor
        generate_random_bid(balance): Returns a random bid between 0 and the total balance of the car
        generate_free_rider_bid(): Returns a bid of 0 (free-riding)
        generate_RL_bid(balance, rush_factor): TODO: Returns a bid, based on the RL bidding strategy. For now, return static bid
        generate_bid(bidding_strategy, balance, rush_factor): Returns a bid, based on the bidding strategy
        ready_for_new_epoch(): Prepares the Bid Generator for the next epoch
    """
    static_bid = 15  # To be adjusted manually.

    def __init__(self,):
        """Initialize the BidGenerator object.
        """

    def __str__(self):
        return f'Bid Generator'

    def generate_static_bid(self, rush_factor):
        """Returns a static bid, multiplied by the rush factor
        Args:
            rush_factor (float): The rush factor of the car.
        Returns:
            float: A static bid, multiplied by the rush factor
        """
        return self.static_bid * rush_factor

    def generate_random_bid(self, balance):
        """Returns a random bid between 0 and the total balance of the car.
        Args:
            balance (float): The balance of the car.
        Returns:
            float: A random bid between 0 and the total balance of the car.    
        """
        return random.uniform(0, balance)

    def generate_free_rider_bid(self):
        """Returns a bid of 0 (free-riding).
        Returns:
            float: 0
        """
        return 0

    def generate_RL_bid(self, balance, rush_factor):
        """TODO: Returns a bid, based on the RL bidding strategy. For now, return static bid
        Args:
            balance (float): The balance of the car.
            rush_factor (float): The rush factor of the car.
        """
        return self.generate_static_bid(rush_factor)

    def generate_bid(self, bidding_strategy, balance, rush_factor):
        """Returns a bid, based on the bidding strategy.
        Args:
            bidding_strategy (string): The bidding strategy of the car.
            balance (float): The balance of the car.
            rush_factor (float): The rush factor of the car.
        """

        if bidding_strategy == 'static':
            return self.generate_static_bid(rush_factor)
        if bidding_strategy == 'random':
            return self.generate_random_bid(balance)
        if bidding_strategy == 'free-rider':
            return self.generate_free_rider_bid()
        if bidding_strategy == 'RL':
            return self.generate_RL_bid(balance, rush_factor)
        else:
            print("ERROR: Invalid bidding strategy: ",
                  bidding_strategy,  ". Returning random bid.")
            return self.generate_random_bid(balance)

    def ready_for_new_epoch(self):
        """Prepares the Bid Generator for the next epoch."""
        # Nothing to update.
        pass
