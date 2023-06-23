"""This module contains the AuctionModifier class, which contains the modofier that is used to modify the auction parameters"""
import random


class AuctionModifier:
    """
    This is the AuctionModifier class. The role of the modifier is to give auction parameters for the next auction. 
    Attributes:
        modifier_type (str): The type of the modifier. (e.g. 'random', 'static', 'spsa')
        intersection_id (str): The id of the intersection for which the modifier is used, or 'same' 
            if the same auction parameters are used everywhere
        grid (Grid): The grid object that contains all intersections and car queues
        spsa_parameters (dict): The parameters used for the SPSA algorithm
    Functions:
        generate_random_parameters: Generates random parameters for the next auction
        generate_static_parameters: Generates static parameters for the next auction
        generate_spsa_parameters: Generates spsa parameters for the next auction
        update_spsa_parameters (last_reward): Updates the SPSA parameters for the next auction
        generate_auction_parameters (last_reward): Calls the appropriate function to generate the auction parameters
        ready_for_new_epoch: Prepares the modifier for the next epoch
    """

    def __init__(self, modifier_type, intersection_id, grid):
        """Initialize the AuctionModifier object
        Args:
            modifier_type (str): The type of the modifier. (Can be: 'Random', 'Adaptive', 'Static')
            intersection_id (str): The id of the intersection for which the modifier is used, or 'all' 
                if the same auction parameters are used everywhere
            grid (Grid): The grid object that contains all intersections and car queues
        """
        self.modifier_type = modifier_type
        self.intersection_id = intersection_id
        self.grid = grid

        # SPSA parameters.
        # Theta parameters are: queue_delay_boost, queue_length_boost, modification_boost_limit_min and modification_boost_limit_max
        self.spsa_parameters = {'theta_params': [0.5, 0.5, 1, 3],
                                'k': 0,
                                'l': 0.005,
                                'c': 0.1,
                                'L': 100,
                                'alpha': 0.602,
                                'gamma': 0.101,
                                # Below are the changing parameters
                                'phase': 'setup',  # The phases are rotating and are: 'setup',
                                # A temp variable to store the parameters to check, either theta+ or theta-
                                'params_to_check': [0, 0, 0, 0],
                                # This is theta+ in the SPSA algorithm, with the posiive pertubation
                                'theta_params_plus': [0, 0, 0, 0],
                                # This is theta- in the SPSA algorithm, with the negative pertubation
                                'theta_params_minus': [0, 0, 0, 0],
                                'l_k': 0,
                                'c_k': 0,
                                'delta_k': [0, 0, 0, 0],
                                # This is F(paramteres with positive pertubation)
                                'f_pos': 0,
                                # This is F(paramteres with negative pertubation)
                                'f_neg': 0
                                }

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

    def generate_spsa_parameters(self, last_reward):
        """Returns parameters for the next auction, based on the SPSA algorithm
        Args:
            last_reward (float): The reward from the last auction, using the parameters in params_to_check
        Returns:
            tuple: A tuple containing the queue delay boost, queue length boost and modification boost limits
        """

        self.update_spsa_parameters(last_reward)

        modification_boost_limit = [0, 0]

        queue_delay_boost, queue_length_boost, modification_boost_limit[
            0], modification_boost_limit[1] = self.spsa_parameters['params_to_check']
        return queue_delay_boost, queue_length_boost, modification_boost_limit

    def update_spsa_parameters(self, last_reward):
        """Updates the SPSA parameters for the next auction
        Args:
            last_reward (float): The reward from the last auction, using the parameters in params_to_check
        """

        #### Setup phase ####
        if self.spsa_parameters['phase'] == 'setup':
            # 1. Increase k by 1
            self.spsa_parameters['k'] += 1
            # 2. Calculate l_k, c_k, delta_k, theta+ & theta-
            self.spsa_parameters['l_k'] = self.spsa_parameters['l'] / \
                (self.spsa_parameters['L'] + self.spsa_parameters['k']
                 )**self.spsa_parameters['alpha']

            self.spsa_parameters['c_k'] = self.spsa_parameters['c'] / \
                (self.spsa_parameters['k'])**self.spsa_parameters['gamma']

            self.spsa_parameters['delta_k'] = [
                random.choice([1, -1]) for i in range(4)]

            self.spsa_parameters['theta_params_plus'] = [self.spsa_parameters['theta_params'][i] + self.spsa_parameters['c_k'] * self.spsa_parameters['delta_k'][i]
                                                         for i in range(4)]
            self.spsa_parameters['theta_params_minus'] = [self.spsa_parameters['theta_params'][i] - self.spsa_parameters['c_k'] * self.spsa_parameters['delta_k'][i]
                                                          for i in range(4)]
            # 3. Set params_to_check to theta+
            self.spsa_parameters['params_to_check'] = self.spsa_parameters['theta_params_plus']
            # 4. Set phase to store_F_Pos
            self.spsa_parameters['phase'] = 'store_F_Pos'

        elif self.spsa_parameters['phase'] == 'store_F_Pos':
            # 1. Store F(theta+)
            self.spsa_parameters['f_pos'] = last_reward
            # 2. Set params_to_check to theta-
            self.spsa_parameters['params_to_check'] = self.spsa_parameters['theta_params_minus']
            # 3. Set phase to store_F_Neg
            self.spsa_parameters['phase'] = 'store_F_Neg'

        elif self.spsa_parameters['phase'] == 'store_F_Neg':
            # 1. Store F(theta-)
            self.spsa_parameters['f_neg'] = last_reward

            # 2. Calculate g_k(theta_k)
            g_k = []
            for i in range(4):
                g_k.append((self.spsa_parameters['f_pos'] - self.spsa_parameters['f_neg']) /
                           (2 * self.spsa_parameters['c_k'] * self.spsa_parameters['delta_k'][i]))

            # 3. Calculate theta_k+1 and set theta_params to theta_k+1
            for i in range(4):
                self.spsa_parameters['theta_params'][i] = self.spsa_parameters['theta_params'][i] + \
                    self.spsa_parameters['l_k'] * g_k[i]
            # 4. Set params_to_check to theta_params
            self.spsa_parameters['params_to_check'] = self.spsa_parameters['theta_params']

            # 5. Clear variables for next iteration
            self.spsa_parameters['theta_params_plus'] = [0, 0, 0, 0]
            self.spsa_parameters['theta_params_minus'] = [0, 0, 0, 0]
            self.spsa_parameters['l_k'] = 0
            self.spsa_parameters['c_k'] = 0
            self.spsa_parameters['delta_k'] = [0, 0, 0, 0]
            self.spsa_parameters['f_pos'] = 0
            self.spsa_parameters['f_neg'] = 0

            # 6. Set phase to setup
            self.spsa_parameters['phase'] = 'setup'

    def generate_auction_parameters(self, last_reward):
        """Returns the auction parameters for the next auction, using the appropriate function depending on the modifier type
        Args:
            last_reward (float): The reward from the last auction, using the parameters in params_to_check
        Returns:
            tuple: A tuple containing the queue delay boost, queue length boost and modification boost limits
        Raises:
            Exception: If the modifier type is invalid
        """
        if self.modifier_type == 'random':
            return self.generate_random_parameters()
        elif self.modifier_type == 'static':
            return self.generate_static_parameters()
        elif self.modifier_type == 'spsa':
            return self.generate_spsa_parameters(last_reward)
        else:
            raise Exception("Invalid Auction Modifier Type")

    def ready_for_new_epoch(self):
        """Prepares the Adaptive Auction Modifier for the next epoch."""
        # Nothing to update.
        pass
