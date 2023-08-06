"""This module contains the AuctionModifier class, which contains the modofier that is used to modify the auction parameters"""
import random
from math import exp
import numpy as np
from sklearn.svm import SVR

# The global below is used to store the best hyperparameters found for the different approaches for action selection
# The structure is as follows:
# Boltzmann: uninformed_score, initial_temperature
# E-greedy_decay: uninformed_score, initial_epsilon
BEST_PARAMETERS_ACTION_SELECTION = {'boltzmann': [0, 0.3],
                                    'e_greedy_decay': [0, 0.3]}

MIN_MAX_DELAY_BOOSTS = [0, 5]


class AuctionModifier:
    """
    This is the AuctionModifier class. The role of the modifier is to give auction parameters for the next auction.
    Attributes:
        args (argparse.Namespace): Arguments parsed from the command line
        intersection_id (str): The id of the intersection foFr which the modifier is used, or 'same'
            if the same auction parameters are used everywhere
        TODO: add stuff
    Functions:
        TODO: add functions
    """

    def __init__(self, args, intersection_id):
        """Initialize the AuctionModifier object
        Args:
            args (argparse.Namespace): Arguments parsed from the command line
            intersection_id (str): The id of the intersection for which the modifier is used, or 'same'
                if the same auction parameters are used everywhere
        """
        self.args = args
        self.intersection_id = intersection_id

        # Present regardless of action selection/update rules.
        self.params_and_expected_rewards = {}
        self.init_params_and_expected_rewards_dict()

        #### Action Selection Dictionaries ####
        # There is also random action selection, but it does not require any parameters.
        # Boltzmann Action Selection parameters
        self.action_selection_boltzmann_params = {}
        # E-greedy with decay Action Selection parameters
        self.action_selection_e_greedy_decay_params = {}

        if (self.args.adaptive_auction_action_selection == "boltzmann"):
            self.init_action_selection_boltzmann_params_dict()
        elif (self.args.adaptive_auction_action_selection == "e_greedy_decay"):
            self.init_action_selection_e_greedy_decay_params_dict()

        #### Update Rule Dictionaries ####
        self.reward_update_svr_params = {}
        if (self.args.adaptive_auction_update_rule == 'svr'):
            self.init_reward_update_svr_params_dict()

    def __str__(self):
        return f'Auction Modifier (intersection {self.intersection_id})'

    def get_parameters_and_valuations_and_counts(self):
        """Returns the parameters, valuations and counts for the adaptive algorithm
        Returns:
            tuple: A tuple containing the parameters, valuations and counts for the adaptive algorithm
        """
        return self.params_and_expected_rewards["possible_param_combs"], self.params_and_expected_rewards["expected_rewards"], self.params_and_expected_rewards["counts"]

# Initialisation Functions
    def init_params_and_expected_rewards_dict(self):
        possible_queue_delay_boosts = []
        # Create all possible parameter combinations based on the level of discretization.
        queue_delay_min_limit, queue_delay_max_limit = MIN_MAX_DELAY_BOOSTS
        for i in range(self.args.adaptive_auction_discretization):
            possible_queue_delay_boosts.append(queue_delay_min_limit + i * (
                queue_delay_max_limit - queue_delay_min_limit) / (self.args.adaptive_auction_discretization - 1))

        possible_param_combs = []
        for queue_delay_boost in possible_queue_delay_boosts:
            possible_param_combs.append(
                [queue_delay_boost])

        # Create the initial counts & expected rewards
        counts = [0] * len(possible_param_combs)
        expected_rewards = [0] * len(possible_param_combs)

        self.params_and_expected_rewards = {'possible_param_combs': possible_param_combs,
                                            'counts': counts,
                                            'expected_rewards': expected_rewards,
                                            'number_of_auctions': 0
                                            }

    def init_action_selection_boltzmann_params_dict(self):
        """Initializes the parameters for the boltzmann action selection
        uninformed_score: The initial score for each parameter combination.
        initial_temperature: The initial temperature used for the algorithm
        temperature_decay: The decay of the temperature after each auction (not epoch, as multiple auctions can happen in an epoch).
        """
        uninformed_score, initial_temperature = BEST_PARAMETERS_ACTION_SELECTION['boltzmann']

        # Set number_of_exploration_epochs to something below num_of_epochs in case exploration should stop before the end of the simulation.
        number_of_exploration_epochs = self.args.num_of_epochs
        # Calcualte the decay so that it is ~0 after number_of_exploration_epochs epochs
        temperature_decay = initial_temperature / number_of_exploration_epochs

        self.action_selection_boltzmann_params = {'temperature_decay': temperature_decay,
                                                  'current_temperature': initial_temperature,
                                                  }

        self.params_and_expected_rewards['expected_rewards'] = [uninformed_score] * len(
            self.params_and_expected_rewards['expected_rewards'])

    def init_action_selection_e_greedy_decay_params_dict(self):
        """Initializes the parameters for the e_greedy_decay action selection
        uninformed_score: The initial score for each parameter combination.
        initial_epsilon: The initial esilon used for the algorithm
        epsilon_decay: The decay of the esilon after each auction (not epoch, as multiple auctions can happen in an epoch).
        """
        uninformed_score, initial_epsilon = BEST_PARAMETERS_ACTION_SELECTION[
            'e_greedy_decay']

        # Set number_of_exploration_epochs to something below num_of_epochs in case exploration should stop before the end of the simulation.
        number_of_exploration_epochs = self.args.num_of_epochs
        # Calcualte the decay so that it is ~0 after number_of_exploration_epochs epochs
        esilon_decay = initial_epsilon / number_of_exploration_epochs

        self.action_selection_e_greedy_decay_params = {'epsilon_decay': esilon_decay,
                                                  'current_epsilon': initial_epsilon,
                                                  }

        self.params_and_expected_rewards['expected_rewards'] = [uninformed_score] * len(
            self.params_and_expected_rewards['expected_rewards'])

    def init_reward_update_svr_params_dict(self):
        self.reward_update_svr_params = {'svr_model': SVR(kernel='rbf'),
                                         'update_interval': 10,
                                         'encountered_data': np.array([], ndmin=2),
                                         'received_rewards': np.array([], ndmin=2)
                                         }
# Update Rule Functions

    def update_expected_rewards(self, last_tried_auction_params, reward):
        if self.args.adaptive_auction_update_rule == 'simple_bandit':
            self.update_expected_rewards_simple_bandit(
                last_tried_auction_params, reward)
        elif self.args.adaptive_auction_update_rule == 'svr':
            self.update_expected_rewards_svr(
                last_tried_auction_params, reward)

    def update_expected_rewards_simple_bandit(self, last_tried_auction_params, reward):
        """Updates the bandit parameters for the simple bandit adaptive algorithm, based on the reward received
        Args:
            last_tried_auction_params (tuple): The parameters that were used for the last auction
            reward (float): The reward received for the last auction
        """
        # Update the counts & average scores for the last tried parameters.
        params_index = self.params_and_expected_rewards['possible_param_combs'].index(
            last_tried_auction_params)
        self.params_and_expected_rewards['expected_rewards'][params_index] = ((self.params_and_expected_rewards['expected_rewards'][params_index] *
                                                                               self.params_and_expected_rewards['counts'][params_index]) + reward) / (self.params_and_expected_rewards['counts'][params_index] + 1)
        self.params_and_expected_rewards['counts'][params_index] += 1

    def update_expected_rewards_svr(self, last_tried_auction_params, reward):
        # First, add the experience and reward to the encountered_data and received_rewards.
        self.reward_update_svr_params['encountered_data'] = np.append(
            self.reward_update_svr_params['encountered_data'], np.array(last_tried_auction_params)).reshape(-1, 1)
        self.reward_update_svr_params['received_rewards'] = np.append(
            self.reward_update_svr_params['received_rewards'], np.array(reward))

        # Then, calculate a new fit and expected reward for each possible parameter value if the update interval is reached.
        if self.params_and_expected_rewards['number_of_auctions'] % self.reward_update_svr_params['update_interval'] == 0:
            self.reward_update_svr_params['svr_model'].fit(
                self.reward_update_svr_params['encountered_data'], self.reward_update_svr_params['received_rewards'].ravel())
            # Update the expected rewards for each possible parameter combination.
            for index, _ in enumerate(self.params_and_expected_rewards['possible_param_combs']):
                self.params_and_expected_rewards['expected_rewards'][index] = self.reward_update_svr_params['svr_model'].predict(
                    np.array(self.params_and_expected_rewards['possible_param_combs'][index], ndmin=2))[0]

        # Update the counts for the last tried parameter
        self.params_and_expected_rewards['counts'][self.params_and_expected_rewards['possible_param_combs'].index(
            last_tried_auction_params)] += 1

# Action Selection/Parameter Generation Functions

    def select_auction_params(self):
        """Returns the auction parameters for the next auction, using the chosen action selection algorithm.
        """
        self.params_and_expected_rewards['number_of_auctions'] += 1

        if self.args.adaptive_auction_action_selection == 'boltzmann':
            chosen_param = self.select_auction_params_boltzmann()

        elif self.args.adaptive_auction_action_selection == 'e_greedy_decay':
            chosen_param = self.select_auction_params_e_greedy_decay()

        elif self.args.adaptive_auction_action_selection == 'random':
            chosen_param = random.choice(
                self.action_selection_boltzmann_params['possible_param_combs'])[0]

        return chosen_param

    def select_auction_params_boltzmann(self):
        """Generates the auction parameters for the next auction, using the Boltzmann algorithm."""
        # First, reduce the temperature based on the decay. Once the temperature is equal to 0, stop decreasing it.
        if (self.action_selection_boltzmann_params['current_temperature'] - self.action_selection_boltzmann_params['temperature_decay'] > 0):
            self.action_selection_boltzmann_params[
                'current_temperature'] -= self.action_selection_boltzmann_params['temperature_decay']

        # Calculate the Boltzmann probabilities.
        boltzmann_probabilities = [
            0] * len(self.params_and_expected_rewards['possible_param_combs'])

        max_score_index = self.params_and_expected_rewards['expected_rewards'].index(
            max(self.params_and_expected_rewards['expected_rewards']))
        temporary_sum = 0
        for index, _ in enumerate(self.params_and_expected_rewards['possible_param_combs']):
            temporary_sum += exp(
                (self.params_and_expected_rewards['expected_rewards'][index] - self.params_and_expected_rewards['expected_rewards'][max_score_index]) / self.action_selection_boltzmann_params['current_temperature'])

        for prob_index, _ in enumerate(boltzmann_probabilities):
            boltzmann_probabilities[prob_index] = (exp(
                (self.params_and_expected_rewards['expected_rewards'][prob_index]-self.params_and_expected_rewards['expected_rewards'][max_score_index]) / self.action_selection_boltzmann_params['current_temperature']))/temporary_sum

        # If any probability is 0, set it to extremely low value, so that we can still generate a random choice.
        for prob_index, _ in enumerate(boltzmann_probabilities):
            if boltzmann_probabilities[prob_index] == 0:
                boltzmann_probabilities[prob_index] = 1e-100

        # Last, choose a parameter set based on the calculated probabilities.
        chosen_params = random.choices(
            self.params_and_expected_rewards['possible_param_combs'], weights=boltzmann_probabilities)

        return chosen_params[0][0]

    def select_auction_params_e_greedy_decay(self):
        """Generates the auction parameters for the next auction, using the e_greedy decay algorithm."""
        # First, reduce the temperature based on the decay. Once the temperature is equal to 0, stop decreasing it.
        if (self.action_selection_e_greedy_decay_params['current_epsilon'] - self.action_selection_e_greedy_decay_params['epsilon_decay'] > 0):
            self.action_selection_e_greedy_decay_params[
                'current_epsilon'] -= self.action_selection_e_greedy_decay_params['epsilon_decay']

        epsilon = self.action_selection_e_greedy_decay_params['current_epsilon']
        # Calculate the e_greedy probabilities.
        e_greedy_probabilities = [
            0] * len(self.params_and_expected_rewards['possible_param_combs'])

        # Find the best parameter combination.
        best_param_comb_index = self.params_and_expected_rewards['expected_rewards'].index(
            max(self.params_and_expected_rewards['expected_rewards']))

        # Set the probability of the best parameter combination to 1 - epsilon.
        e_greedy_probabilities[best_param_comb_index] = 1 - epsilon

        # Set the probability of the rest of the parameter combinations to epsilon.
        for prob_index, _ in enumerate(e_greedy_probabilities):
            if prob_index != best_param_comb_index:
                e_greedy_probabilities[prob_index] = epsilon / \
                    (len(e_greedy_probabilities) - 1)

        # Last, choose a parameter set based on the calculated probabilities.
        chosen_params = random.choices(
            self.params_and_expected_rewards['possible_param_combs'], weights=e_greedy_probabilities)

        return chosen_params[0][0]

# General Functions
    def ready_for_new_epoch(self):
        """Prepares the Auction Modifier for the next epoch."""
        # Nothing to update.
        pass
