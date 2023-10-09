"""This module contains the BidGenerator class, which contains the modofier that is used to modify the bidding behaviour of the cars"""
import random
from math import floor

MIN_MAX_LEARNED_BID_AGGRESSION = [0.5, 5.0]
BEST_PARAMETERS_ACTION_SELECTION = {'e_greedy_exp_decay': [0, 1, 0.995]}
BID_AGGRESSION_DISCRETIZATION = 10


class BidGenerator:
    """
    This is the BidGenerator class. The role of the generator is to generate bids for the cars, based on the bidding strategy.
    Attributes:
        args (argparse.Namespace): Arguments parsed from the command line
    Functions:
        generate_homogeneous_bid(urgency): Returns a homogeneous bid which is the urgency
        generate_random_bid(balance): Returns a random bid between 0 and the total balance of the car
        generate_free_rider_bid(): Returns a bid of 0 (free-riding)
        generate_RL_bid(urgency): TODO: Returns a bid, based on the RL bidding strategy. For now, return homogeneous bid
        generate_bid(balance, urgency): Returns a bid, based on the bidding strategy
        ready_for_new_epoch(): Prepares the Bid Generator for the next epoch
    """

    ### General Functions ###
    def __init__(self, args, bidding_type):
        """Initialize the BidGenerator object.
        Args:
            args (dict): The arguments of the simulation.
        """
        self.args = args
        self.bidding_type = bidding_type

        # Only relevant for RL bidders:
        self.params_and_expected_rewards = {}
        self.action_selection_e_greedy_exp_decay_params = {}
        if self.bidding_type == 'RL':
            self.init_params_and_expected_rewards()
            self.init_e_greedy_exp_decay_params()

    def __str__(self):
        return f"BidGenerator: {self.bidding_type}, type: {self.bidding_type}"

    def generate_bid(self, balance, urgency, bidding_aggression):
        """Returns a bid, based on the bidding strategy.
        Args:
            balance (float): The balance of the car.
            urgency (float): The urgency of the car.
            bidding_aggression (float): The bid aggression of the car.
        Raises:
            Exception: If the bidding strategy is not valid.
        """
        bid = 0
        if self.bidding_type == 'homogeneous':
            # For both, the bid is the urgency
            bid = self.generate_homogeneous_bid(urgency)
        elif self.bidding_type == 'heterogeneous':
            bid = self.generate_heterogeneous_bid(urgency, bidding_aggression)
        elif self.bidding_type == 'random':
            bid = self.generate_random_bid(balance)
        elif self.bidding_type == 'free-rider':
            bid = self.generate_free_rider_bid()
        elif self.bidding_type == 'RL':
            bid = self.generate_RL_bid(urgency, bidding_aggression)
        else:
            raise Exception("ERROR: Invalid bidding strategy: ",
                            self.bidding_type,  ". Returning 0 bid.")

        return floor(bid * 100) / 100

    def init_params_and_expected_rewards(self):
        possible_aggressions = []
        # Create all possible parameter combinations based on the level of discretization.
        aggression_min, aggression_max = MIN_MAX_LEARNED_BID_AGGRESSION
        for i in range(BID_AGGRESSION_DISCRETIZATION):
            possible_aggressions.append(round(aggression_min + i * (
                aggression_max - aggression_min) / (BID_AGGRESSION_DISCRETIZATION - 1), 2))

        # Create the initial counts & expected rewards
        counts = [0] * len(possible_aggressions)
        expected_rewards = [0] * len(possible_aggressions)

        self.params_and_expected_rewards = {'possible_aggressions': possible_aggressions,
                                            'counts': counts,
                                            'expected_rewards': expected_rewards,
                                            }

    def init_e_greedy_exp_decay_params(self):
        """Initializes the parameters for the e_greedy_exp_decay action selection
        uninformed_score: The initial score for each parameter combination.
        initial_epsilon: The initial esilon used for the algorithm
        epsilon_decay: The decay of the esilon after each auction (not epoch, as multiple auctions can happen in an epoch).
        """
        uninformed_score, initial_epsilon, epsilon_decay = BEST_PARAMETERS_ACTION_SELECTION[
            'e_greedy_exp_decay']

        self.action_selection_e_greedy_exp_decay_params = {'epsilon_decay': epsilon_decay,
                                                           'current_epsilon': initial_epsilon,
                                                           }

        self.params_and_expected_rewards['expected_rewards'] = [uninformed_score] * len(
            self.params_and_expected_rewards['expected_rewards'])

    ### Bidding Strategies ###
    def generate_homogeneous_bid(self, urgency):
        """Returns a homogeneous bid, which is the urgency
        Args:
            urgency (float): The urgency of the car.
        Returns:
            float: A homogeneous bid which is the urgency
        """
        return urgency

    def generate_heterogeneous_bid(self, urgency, bidding_aggression):
        """Returns a heterogeneous bid, which is the urgency * bidding_aggression (e.g. urgnecy * 1.24)
        Args:
            urgency (float): The urgency of the car.
            bidding_aggression (float): The bid aggression of the car.
        Returns:
            float: A homogeneous bid which is the urgency
        """
        return urgency * bidding_aggression

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

    ## RL Specific Functions ##
    def generate_RL_bidding_aggression(self):
        """Generates the auction parameters for the next auction, using the e_greedy exponential decay algorithm."""
        # First, reduce the temperature based on the decay. Once the temperature is equal to 0, stop decreasing it.
        self.action_selection_e_greedy_exp_decay_params[
            'current_epsilon'] *= self.action_selection_e_greedy_exp_decay_params['epsilon_decay']
        if (self.action_selection_e_greedy_exp_decay_params['current_epsilon'] < 0.001):
            self.action_selection_e_greedy_exp_decay_params['current_epsilon'] = 0
        epsilon = self.action_selection_e_greedy_exp_decay_params['current_epsilon']
        # Calculate the e_greedy_exp_decay probabilities.
        e_greedy_exp_probabilities = [
            0] * len(self.params_and_expected_rewards['possible_aggressions'])

        # Find the best parameter combination.
        best_param_comb_index = self.params_and_expected_rewards['expected_rewards'].index(
            max(self.params_and_expected_rewards['expected_rewards']))

        # Set the probability of the best parameter combination to 1 - epsilon.
        e_greedy_exp_probabilities[best_param_comb_index] = 1 - epsilon

        # Set the probability of the rest of the parameter combinations to epsilon.
        for prob_index, _ in enumerate(e_greedy_exp_probabilities):
            if prob_index != best_param_comb_index:
                e_greedy_exp_probabilities[prob_index] = epsilon / \
                    (len(e_greedy_exp_probabilities) - 1)

        # Last, choose a parameter set based on the calculated probabilities.
        chosen_params = random.choices(
            self.params_and_expected_rewards['possible_aggressions'], weights=e_greedy_exp_probabilities)

        return chosen_params[0]

    def update_expected_rewards(self, bidding_aggression, reward):
        """Updates the bandit parameters for the simple bandit adaptive algorithm, based on the reward received
        Args:
            reward (float): The reward received for the last trip
        """
        # Update the counts & average scores for the last tried bidding_aggression.
        param_index = self.params_and_expected_rewards['possible_aggressions'].index(
            bidding_aggression)
        self.params_and_expected_rewards['expected_rewards'][param_index] = ((self.params_and_expected_rewards['expected_rewards'][param_index] *
                                                                              self.params_and_expected_rewards['counts'][param_index]) + reward) / (self.params_and_expected_rewards['counts'][param_index] + 1)
        self.params_and_expected_rewards['counts'][param_index] += 1

    def generate_RL_bid(self, urgency, bidding_aggression):
        """TODO: Returns a bid, based on the RL bidding strategy. For now, return homogeneous bid
        Args:
            urgency (float): The urgency of the car.
        """

        return urgency * bidding_aggression

    ### New Epoch Functions ###
    def ready_for_new_epoch(self):
        """Prepares the Bid Generator for the next epoch."""
        # Nothing to update.
        pass
