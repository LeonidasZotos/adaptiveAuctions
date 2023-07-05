"""This module contains the AuctionModifier class, which contains the modofier that is used to modify the auction parameters"""
import random
from math import exp


class AuctionModifier:
    """
    This is the AuctionModifier class. The role of the modifier is to give auction parameters for the next auction.
    Attributes:
        args (argparse.Namespace): Arguments parsed from the command line
        intersection_id (str): The id of the intersection for which the modifier is used, or 'same'
            if the same auction parameters are used everywhere
        grid (Grid): The grid object that contains all intersections and car queues
        bandit_params (dict): The parameters used for the bandit adaptive algorithm, if used.
    Functions:
        init_bandit_params: Initializes the bandit parameters for the bandit adaptive algorithm
        generate_random_parameters: Generates random parameters for the next auction
        generate_static_parameters: Generates static parameters for the next auction
        generate_bandit_parameters: Generates the auction parameters for the next auction, using the bandit adaptive algorithm
        update_bandit_params (last_tried_auction_params, reward): Updates the bandit parameters for the bandit adaptive algorithm, based on the reward received
        generate_auction_parameters (last_reward): Calls the appropriate function to generate the auction parameters
        ready_for_new_epoch: Prepares the modifier for the next epoch
    """

    def __init__(self, args, intersection_id, grid):
        """Initialize the AuctionModifier object
        Args:
            args (argparse.Namespace): Arguments parsed from the command line
            intersection_id (str): The id of the intersection for which the modifier is used, or 'same'
                if the same auction parameters are used everywhere
            grid (Grid): The grid object that contains all intersections and car queues
        """
        self.args = args
        self.intersection_id = intersection_id
        self.grid = grid

        self.bandit_params = {}  # Bandit adaptive parameters

        if args.auction_modifier_type == "bandit":
            self.init_bandit_params()

    def __str__(self):
        return f'Auction Modifier (intersection {self.intersection_id})'

# Random and static adaptive algorithm functions
    def generate_random_parameters(self):
        """Generates random parameters for the next auction
        Returns:
            tuple: A tuple containing the queue delay boost, queue length boost and modification boost max limit
        """
        queue_delay_boost = random.uniform(0, 1)
        queue_length_boost = random.uniform(0, 1)
        # The max limit needs to be larger than the min limit
        modification_boost_max_limit = random.uniform(1, 5)

        return queue_delay_boost, queue_length_boost, modification_boost_max_limit

    def generate_static_parameters(self):
        """Returns static parameters for the next auction
        Returns:
            tuple: A tuple containing the queue delay boost, queue length boost and modification max limits
        """
        queue_delay_boost = 0.5
        queue_length_boost = 0.5
        modification_boost_max_limit = 3  # max multiplier of bid
        return queue_delay_boost, queue_length_boost, modification_boost_max_limit

# Bandit adaptive algorithm functions

    def init_bandit_params(self):
        """Initializes the bandit parameters for the bandit adaptive algorithm. 
        The three main parameters that are adapted are: queue delay boost, queue length boost and boost max limit.
        There are some metaparametrers that are also used:
        level_of_discretization: The level of discretization used for the parameters.
        uninformed_score: The initial score for each parameter combination.
        initial_temperature: The initial temperature used for the algorithm
        temperature_decay: The decay of the temperature after each auction (not epoch, as multiple auctions can happen in an epoch).
        """

        level_of_discretization = 5
        uninformed_score = 1
        initial_temperature = 0.1
        temperature_decay = 0.99

        # Create all possible parameter combinations based on the level of discretization.
        possible_param_combs = []
        for i in range(0, 10):
            for j in range(0, 10):
                for k in range(1, 6):
                    possible_param_combs.append(
                        [i/level_of_discretization, j/level_of_discretization, k])
        # Create the initial counts & average scores.
        counts = [0] * len(possible_param_combs)
        average_scores = [uninformed_score] * len(possible_param_combs)

        # Calculate the Boltzmann probabilities.
        boltzmann_probabilities = [0] * len(possible_param_combs)
        for prob_index, _ in enumerate(boltzmann_probabilities):
            boltzmann_probabilities[prob_index] = round(exp(
                average_scores[prob_index]/initial_temperature), 2)
        sum_of_boltzmann_probabilities = sum(boltzmann_probabilities)
        for prob in boltzmann_probabilities:
            prob = prob/sum_of_boltzmann_probabilities

        self.bandit_params = {'possible_param_combs': possible_param_combs,
                              'temperature_decay': temperature_decay,
                              'counts': counts,
                              'average_scores': average_scores,
                              'current_temperature': initial_temperature
                              }

    def generate_bandit_parameters(self):
        """Returns the auction parameters for the next auction, using the bandit adaptive algorithm."""
        # First, reduce the temperature based on the decay.
        self.bandit_params['current_temperature'] = round(self.bandit_params['current_temperature'] *
                                                          self.bandit_params['temperature_decay'], 3)

        # Then, calculate the Boltzmann probabilities.
        boltzmann_probabilities = [
            0] * len(self.bandit_params['possible_param_combs'])

        for prob_index, _ in enumerate(boltzmann_probabilities):
            try:
                boltzmann_probabilities[prob_index] = round(exp(
                    self.bandit_params['average_scores'][prob_index]/self.bandit_params['current_temperature']), 2)
            except:
                print(
                    "ERROR: Error occured when trying to calculate the Boltzmann probabilities")
                print("attempted to calc: exp(",
                      self.bandit_params['average_scores'][prob_index], "/", self.bandit_params['current_temperature'])

        sum_of_boltzmann_probabilities = sum(boltzmann_probabilities)
        for prob in boltzmann_probabilities:
            prob = prob/sum_of_boltzmann_probabilities

        # Last, choose a parameter combination based on the Boltzmann probabilities.
        chosen_params = random.choices(
            self.bandit_params['possible_param_combs'], weights=boltzmann_probabilities)

        return chosen_params[0][0], chosen_params[0][1], chosen_params[0][2]

    def update_bandit_params(self, last_tried_auction_params, reward):
        """Updates the bandit parameters for the bandit adaptive algorithm, based on the reward received
        Args:
            last_tried_auction_params (tuple): The parameters that were used for the last auction
            reward (float): The reward received for the last auction
        """

        # Update the counts, average scores and Boltzmann probabilities
        params_index = self.bandit_params['possible_param_combs'].index(
            last_tried_auction_params)
        self.bandit_params['counts'][params_index] += 1
        self.bandit_params['average_scores'][params_index] = (self.bandit_params['average_scores'][params_index] * (
            self.bandit_params['counts'][params_index]) + reward) / (self.bandit_params['counts'][params_index] + 1)

# General functions

    def generate_auction_parameters(self):
        """Returns the auction parameters for the next auction, using the appropriate function depending on the modifier type
        Returns:
            tuple: A tuple containing the queue delay boost, queue length boost and modification boost max limit
        Raises:
            Exception: If the modifier type is invalid
        """
        if self.args.auction_modifier_type == 'random':
            return self.generate_random_parameters()
        elif self.args.auction_modifier_type == 'static':
            return self.generate_static_parameters()
        elif self.args.auction_modifier_type == 'bandit':
            return self.generate_bandit_parameters()
        else:
            raise Exception("Invalid Auction Modifier Type")

    def ready_for_new_epoch(self):
        """Prepares the Auction Modifier for the next epoch."""
        # Nothing to update.
        pass
