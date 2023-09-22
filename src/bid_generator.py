"""This module contains the BidGenerator class, which contains the modofier that is used to modify the bidding behaviour of the cars"""
import random
from math import floor


class BidGenerator:
    """
    This is the BidGenerator class. The role of the generator is to generate bids for the cars, based on the bidding strategy.
    Attributes:
        args (argparse.Namespace): Arguments parsed from the command line
    Functions:
        generate_static_bid(urgency): Returns a static bid which is the urgency
        generate_random_bid(balance): Returns a random bid between 0 and the total balance of the car
        generate_free_rider_bid(): Returns a bid of 0 (free-riding)
        generate_RL_bid(balance, urgency): TODO: Returns a bid, based on the RL bidding strategy. For now, return static bid
        generate_bid(bidding_strategy, balance, urgency): Returns a bid, based on the bidding strategy
        ready_for_new_epoch(): Prepares the Bid Generator for the next epoch
    """

    # General Functions
    def __init__(self, args):
        """Initialize the BidGenerator object.
        Args:
            args (dict): The arguments of the simulation.
        """
        self.args = args

    def __str__(self):
        return f'Bid Generator'

    def generate_bid(self, bidding_strategy, balance, urgency, bid_aggression):
        """Returns a bid, based on the bidding strategy.
        Args:
            bidding_strategy (string): The bidding strategy of the car.
            balance (float): The balance of the car.
            urgency (float): The urgency of the car.
            bid_aggression (float): The bid aggression of the car.
        Raises:
            Exception: If the bidding strategy is not valid.
        """
        bid = 0
        if bidding_strategy == 'static':
            # For both, the bid is the urgency
            bid = self.generate_static_bid(urgency)
        elif bidding_strategy == 'aggressive':
            bid = self.generate_aggressive_bid(urgency, bid_aggression)
        elif bidding_strategy == 'random':
            bid = self.generate_random_bid(balance)
        elif bidding_strategy == 'free-rider':
            bid = self.generate_free_rider_bid()
        elif bidding_strategy == 'RL':
            bid = self.generate_RL_bid(balance, urgency)
        else:
            raise Exception("ERROR: Invalid bidding strategy: ",
                            bidding_strategy,  ". Returning 0 bid.")

        return bid

    # Bidding Strategies
    def generate_static_bid(self, urgency):
        """Returns a static bid, which is the urgency
        Args:
            urgency (float): The urgency of the car.
        Returns:
            float: A static bid which is the urgency
        """
        return urgency

    def generate_aggressive_bid(self, urgency, bid_aggression):
        """Returns a aggressive bid, which is the urgency * 1 + bid_aggression (e.g. urgnecy * 1.24)
        Args:
            urgency (float): The urgency of the car.
            bid_aggression (float): The bid aggression of the car.
        Returns:
            float: A static bid which is the urgency
        """
        return urgency * ( 1 + bid_aggression)

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

    def generate_RL_bid(self, balance, urgency):
        """TODO: Returns a bid, based on the RL bidding strategy. For now, return static bid
        Args:
            balance (float): The balance of the car.
            urgency (float): The urgency of the car.
        """
        return self.generate_static_bid(urgency)

    # New Epoch Functions
    def ready_for_new_epoch(self):
        """Prepares the Bid Generator for the next epoch."""
        # Nothing to update.
        pass
