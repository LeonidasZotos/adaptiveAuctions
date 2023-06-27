"""This module contains the AuctionModifier class, which contains the modofier that is used to modify the auction parameters"""
import random


class AuctionModifier:
    """
    This is the AuctionModifier class. The role of the modifier is to give auction parameters for the next auction. 
    Attributes:
        args (argparse.Namespace): Arguments parsed from the command line
        intersection_id (str): The id of the intersection for which the modifier is used, or 'same' 
            if the same auction parameters are used everywhere
        grid (Grid): The grid object that contains all intersections and car queues
    Functions:
        generate_random_parameters: Generates random parameters for the next auction
        generate_static_parameters: Generates static parameters for the next auction
        generate_auction_parameters (last_reward): Calls the appropriate function to generate the auction parameters
        ready_for_new_epoch: Prepares the modifier for the next epoch
    """

    def __init__(self, args, intersection_id, grid):
        """Initialize the AuctionModifier object
        Args:
            args (argparse.Namespace): Arguments parsed from the command line
            intersection_id (str): The id of the intersection for which the modifier is used, or 'all' 
                if the same auction parameters are used everywhere
            grid (Grid): The grid object that contains all intersections and car queues
        """
        self.args = args
        self.intersection_id = intersection_id
        self.grid = grid

    def __str__(self):
        return f'Adaptive Auction Modifier (intersection {self.intersection_id})'

    def generate_random_parameters(self):
        """Generates random parameters for the next auction
        Returns:
            tuple: A tuple containing the queue delay boost, queue length boost and modification boost limits
        """
        queue_delay_boost = random.uniform(0, 1)
        queue_length_boost = random.uniform(0, 1)
        modification_boost_limit = [
            random.uniform(1, 10), random.uniform(1, 10)]
        return queue_delay_boost, queue_length_boost, modification_boost_limit

    def generate_static_parameters(self):
        """Returns static parameters for the next auction
        Returns:
            tuple: A tuple containing the queue delay boost, queue length boost and modification boost limits
        """
        queue_delay_boost = 0.5
        queue_length_boost = 0.5
        modification_boost_limit = [1, 3]  # min/max multiplier of bid
        return queue_delay_boost, queue_length_boost, modification_boost_limit

    def generate_adaptive_parameters(self):
        raise Exception("Not implemented yet")
        pass
        

    def generate_auction_parameters(self, last_reward):
        """Returns the auction parameters for the next auction, using the appropriate function depending on the modifier type
        Args:
            last_reward (float): The reward from the last auction, using the parameters in params_to_check
        Returns:
            tuple: A tuple containing the queue delay boost, queue length boost and modification boost limits
        Raises:
            Exception: If the modifier type is invalid
        """
        if self.args.auction_modifier_type == 'random':
            return self.generate_random_parameters()
        elif self.args.auction_modifier_type == 'static':
            return self.generate_static_parameters()
        elif self.args.auction_modifier_type == 'adaptive':
            return self.generate_adaptive_parameters(last_reward)
        else:
            raise Exception("Invalid Auction Modifier Type")

    def ready_for_new_epoch(self):
        """Prepares the Adaptive Auction Modifier for the next epoch."""
        # Nothing to update.
        pass
