"""This builds on 8 and introduces adaptive auctions again. Auctions train on max_time_waited, and bids are boosted by an inactivity factor that the auction determines."""
import os
import numpy as np
import random
from math import exp, inf
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


def plot_bandit_valuations(auctionModifier):
    params = auctionModifier.bandit_params["possible_boosts"]
    valuations = auctionModifier.bandit_params["average_scores"]
    counts = auctionModifier.bandit_params["counts"]
    plt.scatter(params, valuations, s=counts, color='black')
    plt.xlabel("Inactivity boost")
    plt.ylabel("Average bandit valuation")

    if not os.path.exists(results_folder_name):
        os.makedirs(results_folder_name)
    plt.savefig(results_folder_name + '/bandit_valuations.png')
    plt.close()


def plot_average_max_time_waited_per_boost(results_folder_name, results, counts):
    # Counts are divided by 2, so that the size is not too big
    x = np.array([result[0] for result in results])
    y = np.array([result[1] for result in results])

    # for each x, calculate the average y value
    x_unique = np.unique(x)
    y_unique = np.array([np.mean(y[x == i]) for i in x_unique])
    # The size of each marker is proportional to the number of times that boost was used
    plt.scatter(x_unique, y_unique, s=counts, color='black')
    plt.xlabel("Inactivity boost")
    plt.ylabel("1/max_time_waited")

    if not os.path.exists(results_folder_name):
        os.makedirs(results_folder_name)
    plt.savefig(results_folder_name + '/average_revenue_per_boost.png')
    plt.close()


def plot_metric_over_time(results_folder_name, revenues_adaptive, revenues_random, variable_name, exclude_first_x = 20):
    import matplotlib.pyplot as plt
    import numpy as np

    # Calculate the average revenue for each auction
    x = np.array([i for i in range(len(revenues_adaptive[0]))])
    y_adaptive = np.array([np.mean([revenues_adaptive[i][j] for i in range(len(revenues_adaptive))])
                           for j in range(len(revenues_adaptive[0]))])
    y_random = np.array([np.mean([revenues_random[i][j] for i in range(len(revenues_random))])
                         for j in range(len(revenues_random[0]))])
    # Exclude the first x auctions, as they are not representative
    x = x[exclude_first_x:]
    y_adaptive = y_adaptive[exclude_first_x:]
    y_random = y_random[exclude_first_x:]

    plt.plot(x, y_adaptive, label="adaptive")
    plt.plot(x, y_random, label="random")
    plt.xlabel("Auction number")
    plt.ylabel(variable_name)
    plt.legend()

    if not os.path.exists(results_folder_name):
        os.makedirs(results_folder_name)
    plt.savefig(results_folder_name + '/' +
                str(variable_name) + '_over_time.png')
    plt.close()



class AuctionModifier:
    def __init__(self, inactivity_boost_min_max, num_of_auctions):
        self.num_of_auctions = num_of_auctions
        self.discretization = 13
        # possible inactivity_boosts are between inactivity_boost_min_max[0] and inactivity_boost_min_max[1], spaced out evenly
        self.inactivity_boosts = list(np.linspace(
            inactivity_boost_min_max[0], inactivity_boost_min_max[1], self.discretization))

        self.bandit_params = {}   # Bandit adaptive parameters
        self.init_bandit_params()

    def get_counts(self):
        return self.bandit_params['counts']

    def init_bandit_params(self):
        uninformed_score = 0
        initial_temperature = 0.2 # was 0.1
        final_temperature = 0.05 # was 0.01
        # calculate the decay needed to reach the final temperature after num_of_auctions auctions
        temperature_decay = (initial_temperature -
                             final_temperature) / self.num_of_auctions
        counts = [1] * len(self.inactivity_boosts)
        average_scores = [uninformed_score] * len(self.inactivity_boosts)

        self.bandit_params = {'possible_boosts': self.inactivity_boosts,
                              'temperature_decay': temperature_decay,
                              'counts': counts,
                              'average_scores': average_scores,
                              'current_temperature': initial_temperature
                              }

    def generate_inactivity_boost(self):
        """Returns the possible_boosts for the next auction, using the bandit adaptive algorithm."""
        # First, reduce the temperature
        self.bandit_params['current_temperature'] = self.bandit_params['current_temperature'] - \
            self.bandit_params['temperature_decay']

        # Then, calculate the Boltzmann probabilities.
        boltzmann_probabilities = [
            0] * len(self.bandit_params['possible_boosts'])

        for prob_index, _ in enumerate(boltzmann_probabilities):
            try:
                boltzmann_probabilities[prob_index] = exp(
                    self.bandit_params['average_scores'][prob_index]/self.bandit_params['current_temperature'])
            except OverflowError:
                boltzmann_probabilities[prob_index] = inf

        sum_of_boltzmann_probabilities = sum(boltzmann_probabilities)
        # divide each probability by the sum of all probabilities
        for prob_index, _ in enumerate(boltzmann_probabilities):
            boltzmann_probabilities[prob_index] /= sum_of_boltzmann_probabilities

        # Last, choose an inactivity boost based on the Boltzmann probabilities.
        chosen_inactivity_bost = random.choices(
            self.bandit_params['possible_boosts'], weights=boltzmann_probabilities)

        # If there is a parameter with count 1, choose that one, so that we try it out.
        if 1 in self.bandit_params['counts']:
            chosen_inactivity_bost = [
                self.bandit_params['possible_boosts'][self.bandit_params['counts'].index(1)]]

        return chosen_inactivity_bost[0]

    def update_bandit_valuations(self, inactivity_boost, training_metric):
        params_index = self.bandit_params['possible_boosts'].index(
            inactivity_boost)
        self.bandit_params['counts'][params_index] += 1
        self.bandit_params['average_scores'][params_index] = (self.bandit_params['average_scores'][params_index] * (
            self.bandit_params['counts'][params_index]) + training_metric) / (self.bandit_params['counts'][params_index] + 1)


class Bidder:
    def __init__(self, valuation):
        # Will be different for the competitor, as in Pardoe 2006
        self.valuation = valuation
        # This is set when the bidder enters an auction and is affected by their time since the last win
        self.boosted_valuation = 0
        self.time_since_last_win = 0

    def set_boosted_valuation(self, inactivity_boost):
        self.boosted_valuation = self.valuation + \
            (self.time_since_last_win * inactivity_boost)

    def get_native_valuation(self):
        return self.valuation

    def get_full_boosted_bid(self):
        # Is this a valid shortcut to having to go through the whole increments? That will save computational time
        return self.boosted_valuation


class Auction:
    def __init__(self, inactivity_boost, bidders):
        self.inactivity_boost = inactivity_boost
        self.bidders = bidders
        self.increment = 0.01
        self.revenue = 0
        self.winner = None

    def get_end_of_auction_stats(self):
        max_time_waited = 1 / \
            (max([bidder.time_since_last_win for bidder in self.bidders]))
        return self.inactivity_boost, self.revenue, max_time_waited

    def get_winner(self):
        return self.winner

    def run_auction(self):
        highest_bid_holder = None
        # Randomly order the bidders
        # Everyone participates since the reserve price is 0
        random.shuffle(self.bidders)
        for bidder in self.bidders:
            bidder.set_boosted_valuation(self.inactivity_boost)

        # Gather all (boosted) bids. # This is a shortcut. Instead of going through all the increments, we know that the winner will pay the 2nd highest bid + 0.01
        boosted_bids = [bidder.get_full_boosted_bid()
                        for bidder in self.bidders]
        # Order bids and bidders by bid. Bids are in decreasing order
        boosted_bids, self.bidders = zip(
            *sorted(zip(boosted_bids, self.bidders), reverse=True))
        highest_bid_holder = self.bidders[0]
        second_highest_bid = self.bidders[1].get_full_boosted_bid()
        # At this stage, there is only one bidder left, and they are the winner
        self.revenue = second_highest_bid + 0.01
        self.winner = highest_bid_holder

        self.winner.time_since_last_win = 0

        for bidder in self.bidders:
            # Increase time waited since last win.
            bidder.time_since_last_win += 1


def create_bidders(num_of_bidders, valuations_min_max):
    # create gaussian distribution of valuations, with mean randomly picked between 0 and 1 and variance 10^x where x is randomly picked between -2 and 1
    all_bidders = []

    for bidder in range(num_of_bidders):
        # Every bidder comes from a different distribution
        mu_v, sigma_v = np.random.uniform(
            valuations_min_max[0], valuations_min_max[1]), 10**np.random.uniform(-2, 1)
        valuation = np.random.normal(mu_v, sigma_v)
        while valuation < valuations_min_max[0] or valuation > valuations_min_max[1]:
            valuation = np.random.normal(mu_v, sigma_v)

        all_bidders.append(Bidder(valuation))

    return all_bidders


def run_simulation(reserve):
    inactivity_boost_min_max = [0, 10]
    total_number_of_auctions = 2000
    valuations_min_max = [0, 1]
    boosts_revenues_times = []  # Holds the auction results
    revenues_over_time = []  # Holds the revenue for each auction
    max_time_waited_over_time = []
    auction_modifier = AuctionModifier(
        inactivity_boost_min_max, total_number_of_auctions)

    bidders = create_bidders(random.randint(2, 4), valuations_min_max)
    for auction_id in range(total_number_of_auctions):
        inactivity_boost = 0
        if reserve == 'adaptive':
            inactivity_boost = auction_modifier.generate_inactivity_boost()
        elif reserve == 'random':
            inactivity_boost = random.choices(list(np.linspace(
                inactivity_boost_min_max[0], inactivity_boost_min_max[1], 13)))[0]  # randomly pick a inactivity_boost

        auction = Auction(inactivity_boost, bidders)
        auction.run_auction()
        boosts_revenues_times.append(auction.get_end_of_auction_stats())
        revenues_over_time.append(auction.get_end_of_auction_stats()[1])
        max_time_waited_over_time.append(auction.get_end_of_auction_stats()[2])

        # Update the reserve price for the next auction
        auction_modifier.update_bandit_valuations(auction.get_end_of_auction_stats()[
            0], auction.get_end_of_auction_stats()[2])  # We train on max_time_waited

        # Remove winner and potentially add new bidders
        bidders.remove(auction.get_winner())
        num_of_spots = 4 - len(bidders)
        if num_of_spots == 3:  # If there are 3 spots, create at least 1 new bidder
            new_bidders = create_bidders(random.randint(
                1, num_of_spots), valuations_min_max)
        else:  # Otherwise, there is also a chance to create 0 new bidders
            new_bidders = create_bidders(random.randint(
                0, num_of_spots), valuations_min_max)

        bidders.extend(new_bidders)

    return boosts_revenues_times, revenues_over_time, max_time_waited_over_time, auction_modifier


if __name__ == '__main__':
    results_folder_name = str(os.path.basename(__file__))
    results_folder_name = results_folder_name[:-3]
    revenues_over_time_all_sims_adaptive = []
    max_time_waited_over_time_all_sims_adaptive = []
    revenues_over_time_all_sims_random = []
    max_time_waited_over_time_all_sims_random = []
    last_sim_boosts_and_revenues = []
    last_auction_modifier = None
    num_of_sims = 2000

    pool = Pool()  # Default number of processes will be used

    with tqdm(total=num_of_sims) as pbar:
        for results in pool.imap(run_simulation, ["adaptive"] * num_of_sims):
            last_sim_boosts_and_revenues = results[0]
            last_auction_modifier = results[3]
            revenues_over_time_all_sims_adaptive.append(results[1])
            max_time_waited_over_time_all_sims_adaptive.append(results[2])
            pbar.update()

    plot_bandit_valuations(last_auction_modifier)

    plot_average_max_time_waited_per_boost(results_folder_name,
                                           last_sim_boosts_and_revenues, last_auction_modifier.get_counts())

    with tqdm(total=num_of_sims) as pbar:
        for results in pool.imap(run_simulation, ["random"] * num_of_sims):
            last_sim_boosts_and_revenues = results[0]
            last_auction_modifier = results[3]
            revenues_over_time_all_sims_random.append(results[1])
            max_time_waited_over_time_all_sims_random.append(results[2])
            pbar.update()

    pool.close()
    pool.join()

    # First, we plot the revenues over time
    plot_metric_over_time(results_folder_name,
                          revenues_over_time_all_sims_adaptive, revenues_over_time_all_sims_random, "Revenue")

    # Then, we plot the 1/(max_time_waited) over time
    plot_metric_over_time(results_folder_name,
                          max_time_waited_over_time_all_sims_adaptive, max_time_waited_over_time_all_sims_random, "1over1+max_time_waited")
