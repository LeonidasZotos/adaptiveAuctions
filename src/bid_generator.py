"""This module contains the BidGenerator class, which contains the modofier that is used to modify the bidding behaviour of the cars"""
import random


class BidGenerator:
    """
    This is the BidGenerator class. The role of the generator is to generate bids for the cars, based on the bidding strategy.
    Attributes:
        args (argparse.Namespace): Arguments parsed from the command line
    Functions:
        generate_static_bid(urgency): Returns a static bid, multiplied by the urgency
        generate_random_bid(balance): Returns a random bid between 0 and the total balance of the car
        generate_free_rider_bid(): Returns a bid of 0 (free-riding)
        generate_RL_bid(balance, urgency): TODO: Returns a bid, based on the RL bidding strategy. For now, return static bid
        generate_bid(bidding_strategy, balance, urgency): Returns a bid, based on the bidding strategy
        ready_for_new_epoch(): Prepares the Bid Generator for the next epoch
    """
    static_bid = 5  # This needs to be adjusted manually.

    def __init__(self, args):
        """Initialize the BidGenerator object.
        Args:
            args (dict): The arguments of the simulation.
        """
        self.args = args

    def __str__(self):
        return f'Bid Generator'

    def generate_static_bid(self, urgency):
        """Returns a static bid, multiplied by the urgency
        Args:
            urgency (float): The urgency of the car.
        Returns:
            float: A static bid, multiplied by the urgency
        """
        return self.static_bid * urgency

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

    def generate_bid(self, bidding_strategy, balance, urgency):
        """Returns a bid, based on the bidding strategy.
        Args:
            bidding_strategy (string): The bidding strategy of the car.
            balance (float): The balance of the car.
            urgency (float): The urgency of the car.
        Raises:
            Exception: If the bidding strategy is not valid.
        """

        if bidding_strategy == 'static':
            return self.generate_static_bid(urgency)
        if bidding_strategy == 'random':
            return self.generate_random_bid(balance)
        if bidding_strategy == 'free-rider':
            return self.generate_free_rider_bid()
        if bidding_strategy == 'RL':
            return self.generate_RL_bid(balance, urgency)
        else:
            raise Exception("ERROR: Invalid bidding strategy: ",
                            bidding_strategy,  ". Returning random bid.")

    def ready_for_new_epoch(self):
        """Prepares the Bid Generator for the next epoch."""
        # Nothing to update.
        pass
