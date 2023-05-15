"""This module contains the AuctionModifier class, which contains the modofier that is used to modify the auction parameters"""
import random


class AuctionModifier:
    """
    This is the AuctionModifier class. The role of the modifier is to give auction parameters for the next auction. 
    Plan on creating static and adaptive modifiers.
    For now, it returns random paramters.
    Attributes:
        intersection_id (str): The id of the intersection for which the modifier is used, or 'all' 
            if the same auction parameters are used everywhere
        modifier_type (str): The type of the modifier. (e.g. 'random', 'static', 'spsa')
    Functions:
        ready_for_new_epoch: Prepares the modifier for the next epoch.
        generate_auction_parameters: Calls the appropriate function to generate the auction parameters
        generate_random_parameters: Generates random parameters for the next auction
        generate_static_parameters: Generates static parameters for the next auction
        generate_spsa_parameters: Generates spsa parameters for the next auction

    """

    def __init__(self, modifier_type, intersection_id):
        """Initialize the AuctionModifier object
        Args:
            modifier_type (str): The type of the modifier. (Can be: 'Random', 'Adaptive', 'Static')
            intersection_id (str): The id of the intersection for which the modifier is used, or 'all' 
                if the same auction parameters are used everywhere
        """
        self.intersection_id = intersection_id
        self.modifier_type = modifier_type

    def __str__(self):
        return f'Adaptive Auction Modifier'

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
        modification_boost_limit = [1, 2]
        return queue_delay_boost, queue_length_boost, modification_boost_limit

    def generate_spsa_parameters(self):
        """Returns parameters for the next auction, based on the SPSA algorithm
        Returns:
            tuple: A tuple containing the queue delay boost, queue length boost and modification boost limits
        """
        print("SPSA ALGORITHM IS NOT IMPLEMENTED YET, returning static parameters")
        queue_delay_boost = 0.5
        queue_length_boost = 0.5
        modification_boost_limit = [1, 2]
        return queue_delay_boost, queue_length_boost, modification_boost_limit

    def generate_auction_parameters(self):
        """Returns the auction parameters for the next auction, using the appropriate function depending on the modifier type
        Returns:
            tuple: A tuple containing the queue delay boost, queue length boost and modification boost limits
        """
        if self.modifier_type == 'random':
            return self.generate_random_parameters()
        elif self.modifier_type == 'static':
            return self.generate_static_parameters()
        elif self.modifier_type == 'spsa':
            return self.generate_spsa_parameters()
        else:
            print("ERROR: Invalid Auction Modifier Type, using random auction parameters")
            return self.generate_random_parameters()

    def ready_for_new_epoch(self):
        """Prepares the Adaptive Auction Modifier for the next epoch."""
        # Nothing to update.
        pass
