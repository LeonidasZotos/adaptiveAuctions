"""This module contains the AuctionModifier class, which contains the modofier that is used to modify the auction parameters"""
import random
from math import exp

# The global below is used to store the best hyperparameters found for the different approaches for action selection
# The structure is as follows:
# Boltzmann: uninformed_score, initial_temperature
# E-greedy_decay: uninformed_score, initial_temperature
BEST_PARAMETERS_ACTION_SELECTION = {'boltzmann': [0, 0.3],
                                    'e-greedy_decay': [0, 0.3],
                                    'random': [0, 0]}


class AuctionModifier:
    """
    This is the AuctionModifier class. The role of the modifier is to give auction parameters for the next auction.
    Attributes:
        args (argparse.Namespace): Arguments parsed from the command line
        intersection_id (str): The id of the intersection for which the modifier is used, or 'same'
            if the same auction parameters are used everywhere
        grid (Grid): The grid object that contains all intersections and car queues
        simple_bandit_params (dict): The parameters used for the simple bandit adaptive algorithm, if used.
    Functions:
        get_parameters_and_valuations_and_counts: Returns the parameters, valuations and counts for the adaptive algorithm
        init_simple_bandit_params: Initializes the bandit parameters for the bandit adaptive algorithm
        generate_random_parameters: Generates random parameters for the next auction
        generate_static_parameters: Generates static parameters for the next auction
        generate_simple_bandit_parameters: Returns the auction parameters for the next auction, using the bandit adaptive algorithm.
        update_params_simple_bandit (last_tried_auction_params, reward): Updates the bandit parameters for the bandit adaptive algorithm, based on the reward received
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

        self.simple_bandit_params = {}  # Bandit adaptive parameters

        if args.adaptive_auction_update_rule == "simple_bandit":
            self.init_simple_bandit_params()

    def __str__(self):
        return f'Auction Modifier (intersection {self.intersection_id})'

    def get_parameters_and_valuations_and_counts(self):
        """Returns the parameters, valuations and counts for the adaptive algorithm
        Returns:
            None or
            tuple: A tuple containing the parameters, valuations and counts for the adaptive algorithm
        """
        if self.args.adaptive_auction_update_rule == 'simple_bandit':
            return self.simple_bandit_params["possible_param_combs"], self.simple_bandit_params["average_scores"], self.simple_bandit_params["counts"]
        return None

# Random and static adaptive algorithm functions
    def generate_random_parameters(self):
        """Generates random parameters for the next auction
        Returns:
            tuple: A tuple containing the queue delay boost. Potential to add more params.
        """
        queue_delay_boost = random.uniform(0, 1)

        return queue_delay_boost

    def generate_static_parameters(self):
        """Returns static parameters for the next auction
        Returns:
            tuple: A tuple containing the queue delay boost. Potential to add more params.
        """
        queue_delay_boost = 0.5
        return queue_delay_boost

# Bandit adaptive algorithm functions
    def init_simple_bandit_params(self):
        """Initializes the bandit parameters for the simple_bandit adaptive algorithm.
        The param is for now: queue delay boost.
        There are some metaparametrers that are also used:
        uninformed_score: The initial score for each parameter combination.
        initial_temperature: The initial temperature used for the algorithm
        temperature_decay: The decay of the temperature after each auction (not epoch, as multiple auctions can happen in an epoch).
        """

        # number of values to try for each parameter
        level_of_discretization = self.args.adaptive_auction_discretization
        uninformed_score = 0
        initial_temperature = 0
        if self.args.adaptive_auction_action_selection == 'boltzmann':
            uninformed_score, initial_temperature = BEST_PARAMETERS_ACTION_SELECTION[
                'boltzmann']
        elif self.args.adaptive_auction_action_selection == 'e-greedy-decay':
            uninformed_score, initial_temperature = BEST_PARAMETERS_ACTION_SELECTION[
                'e-greedy_decay']

        # Set number_of_exploration_epochs to something below num_of_epochs in case exploration should stop before the end of the simulation.
        number_of_exploration_epochs = self.args.num_of_epochs
        # Calcualte the decay so that it is ~0 after number_of_exploration_epochs epochs
        temperature_decay = initial_temperature / number_of_exploration_epochs

        # Create all possible parameter combinations based on the level of discretization.
        queue_delay_min_limit = 0
        queue_delay_max_limit = 5

        possible_queue_delay_boosts = []

        for i in range(level_of_discretization):
            possible_queue_delay_boosts.append(queue_delay_min_limit + i * (
                queue_delay_max_limit - queue_delay_min_limit) / (level_of_discretization - 1))

        possible_param_combs = []
        for queue_delay_boost in possible_queue_delay_boosts:
            possible_param_combs.append(
                [queue_delay_boost])

        # Create the initial counts & average scores.
        counts = [0] * len(possible_param_combs)
        average_scores = [uninformed_score] * len(possible_param_combs)

        self.simple_bandit_params = {'possible_param_combs': possible_param_combs,
                              'temperature_decay': temperature_decay,
                              'counts': counts,
                              'average_scores': average_scores,
                              'current_temperature': initial_temperature,
                              'number_of_auctions': 0
                              }

    def select_params_boltzmann(self):
        """Generates the auction parameters for the next auction, using the Boltzmann algorithm."""
        # First, reduce the temperature based on the decay. Once the temperature is equal to 0, stop decreasing it.
        if (self.simple_bandit_params['current_temperature'] - self.simple_bandit_params['temperature_decay'] > 0):
            self.simple_bandit_params['current_temperature'] -= self.simple_bandit_params['temperature_decay']

        # Calculate the Boltzmann probabilities.
        boltzmann_probabilities = [
            0] * len(self.simple_bandit_params['possible_param_combs'])

        max_score_index = self.simple_bandit_params['average_scores'].index(
            max(self.simple_bandit_params['average_scores']))
        temporary_sum = 0
        for index, _ in enumerate(self.simple_bandit_params['possible_param_combs']):
            temporary_sum += exp(
                (self.simple_bandit_params['average_scores'][index] - self.simple_bandit_params['average_scores'][max_score_index]) / self.simple_bandit_params['current_temperature'])

        for prob_index, _ in enumerate(boltzmann_probabilities):
            boltzmann_probabilities[prob_index] = (exp(
                (self.simple_bandit_params['average_scores'][prob_index]-self.simple_bandit_params['average_scores'][max_score_index]) / self.simple_bandit_params['current_temperature']))/temporary_sum

        # If any probability is 0, set it to extremely low value, so that we can still generate a random choice.
        for prob_index, _ in enumerate(boltzmann_probabilities):
            if boltzmann_probabilities[prob_index] == 0:
                boltzmann_probabilities[prob_index] = 1e-100
        # Round to 2 s.f.
        final_probabilities = [round(elem, 2)
                               for elem in boltzmann_probabilities]

        # Last, choose a parameter set based on the calculated probabilities.
        chosen_params = random.choices(
            self.simple_bandit_params['possible_param_combs'], weights=final_probabilities)

        return chosen_params[0][0]

    def select_params_e_greedy_decay(self):
        """Generates the auction parameters for the next auction, using the e-greedy decay algorithm."""
        # First, reduce the temperature based on the decay. Once the temperature is equal to 0, stop decreasing it.
        if (self.simple_bandit_params['current_temperature'] - self.simple_bandit_params['temperature_decay'] > 0):
            self.simple_bandit_params['current_temperature'] -= self.simple_bandit_params['temperature_decay']

        epsilon = self.simple_bandit_params['current_temperature']
        # Calculate the e-greedy probabilities.
        e_greedy_probabilities = [
            0] * len(self.simple_bandit_params['possible_param_combs'])

        # Find the best parameter combination.
        best_param_comb_index = self.simple_bandit_params['average_scores'].index(
            max(self.simple_bandit_params['average_scores']))

        # Set the probability of the best parameter combination to 1 - epsilon.
        e_greedy_probabilities[best_param_comb_index] = 1 - epsilon

        # Set the probability of the rest of the parameter combinations to epsilon.
        for prob_index, _ in enumerate(e_greedy_probabilities):
            if prob_index != best_param_comb_index:
                e_greedy_probabilities[prob_index] = epsilon / \
                    (len(e_greedy_probabilities) - 1)

        # Last, choose a parameter set based on the calculated probabilities.
        chosen_params = random.choices(
            self.simple_bandit_params['possible_param_combs'], weights=e_greedy_probabilities)

        return chosen_params[0][0]

    def select_auction_params(self):
        """Returns the auction parameters for the next auction, using the chosen algorithm.
        """
        self.simple_bandit_params['number_of_auctions'] += 1

        if self.args.adaptive_auction_action_selection == 'boltzmann':
            chosen_param = self.select_params_boltzmann()

        elif self.args.adaptive_auction_action_selection == 'e-greedy_decay':
            chosen_param = self.select_params_e_greedy_decay()

        elif self.args.adaptive_auction_action_selection == 'random':
            chosen_param = random.choice(
                self.simple_bandit_params['possible_param_combs'])[0]

        return chosen_param

    def update_params_simple_bandit(self, last_tried_auction_params, reward):
        """Updates the bandit parameters for the bandit adaptive algorithm, based on the reward received
        Args:
            last_tried_auction_params (tuple): The parameters that were used for the last auction
            reward (float): The reward received for the last auction
        """
        # Update the counts, average scores and Boltzmann probabilities
        params_index = self.simple_bandit_params['possible_param_combs'].index(
            last_tried_auction_params)
        self.simple_bandit_params['average_scores'][params_index] = ((self.simple_bandit_params['average_scores'][params_index] *
                                                               self.simple_bandit_params['counts'][params_index]) + reward) / (self.simple_bandit_params['counts'][params_index] + 1)
        self.simple_bandit_params['counts'][params_index] += 1

# General functions
    def generate_auction_parameters(self):
        """Returns the auction parameters for the next auction, using the appropriate function depending on the modifier type
        Returns:
            tuple: A tuple containing the queue delay boost. Potential to add more params.
        Raises:
            Exception: If the modifier type is invalid
        """
        if self.args.adaptive_auction_update_rule == 'random':
            return self.generate_random_parameters()
        elif self.args.adaptive_auction_update_rule == 'static':
            return self.generate_static_parameters()
        elif self.args.adaptive_auction_update_rule == 'simple_bandit':
            return self.select_auction_params()
        else:
            raise Exception("Invalid Auction Modifier Type")

    def ready_for_new_epoch(self):
        """Prepares the Auction Modifier for the next epoch."""
        # Nothing to update.
        pass
