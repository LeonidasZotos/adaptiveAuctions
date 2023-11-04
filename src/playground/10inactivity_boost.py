"""This builds on 9 and introduces adaptive auctions again. Auctions train on inact_rank, and bids are boosted by an inactivity factor that the auction determines."""
import os
import numpy as np
import random
from math import exp, inf
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


def plot_average_bandit_valuations(auction_modifiers):
    num_of_modifiers = len(auction_modifiers)
    params = auction_modifiers[0].bandit_params["possible_boosts"]
    average_valuations_per_boost = [0] * len(params)
    counts = [0] * len(params)

    for modifier in auction_modifiers:
        for index, param in enumerate(params):
            average_valuations_per_boost[index] += modifier.bandit_params["average_scores"][index]
            counts[index] += modifier.bandit_params["counts"][index]

    for index, _ in enumerate(average_valuations_per_boost):
        average_valuations_per_boost[index] /= num_of_modifiers

    # Counts are divided by num_of_modifiers, so that we take the average
    counts = [count/num_of_modifiers for count in counts]

    plt.scatter(params, average_valuations_per_boost, s=counts, color='black')
    plt.xlabel("Inactivity boost")
    plt.ylabel("Average bandit valuation")

    if not os.path.exists(results_folder_name):
        os.makedirs(results_folder_name)
    plt.savefig(results_folder_name + '/bandit_valuations.png')
    plt.close()


def plot_metric_over_time(results_folder_name, metrics_adaptive, metrics_random, variable_name, exclude_first_x=13):
    # We exclude the first x trials because during that time, the bandit algorithm is just trying out all x options for the 1st time.
    # Calculate the average metric value for each auction
    x = list(range(len(metrics_adaptive[0])))
    y_adaptive = [0] * len(metrics_adaptive[0])
    y_random = [0] * len(metrics_random[0])

    for sim in metrics_adaptive:
        for epoch in range(len(sim)):
            y_adaptive[epoch] += sim[epoch][1]

    y_adaptive = [metric/len(metrics_adaptive) for metric in y_adaptive]

    for sim in metrics_random:
        for epoch in range(len(sim)):
            y_random[epoch] += sim[epoch][1]

    y_random = [metric/len(metrics_random) for metric in y_random]

    # Exclude the first x auctions, as they are not representative
    x = x[exclude_first_x:]
    y_adaptive = y_adaptive[exclude_first_x:]
    y_random = y_random[exclude_first_x:]

    plt.plot(x, y_adaptive, label="Adaptive")
    plt.plot(x, y_random, label="Random")
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
        initial_temperature = 0.1  # was 0.1
        final_temperature = 0.01  # was 0.01
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
        # Time since the last win
        self.time_since_last_win = 0

        # Auction-dependent parameters. Set at the start of each auction.
        self.boosted_bid = 0
        self.inact_rank = 0

    def get_valuation(self):
        return self.valuation

    def get_time_since_last_win(self):
        return self.time_since_last_win

    def increase_time_since_last_win(self):
        self.time_since_last_win += 1

    def set_boosted_bid(self, boosted_bid):
        self.boosted_bid = boosted_bid

    def get_boosted_bid(self):
        return self.boosted_bid

    def set_inact_rank(self, rank):
        self.inact_rank = rank

    def get_inact_rank(self):
        return self.inact_rank


class Auction:
    def __init__(self, inactivity_boost, bidders):
        self.inactivity_boost = inactivity_boost
        self.bidders = bidders
        self.increment = 0.01
        self.revenue = 0
        self.winner = None

    def get_end_of_auction_stats(self):
        winner_time_rank = self.winner.get_inact_rank()
        return self.inactivity_boost, winner_time_rank

    def get_winner(self):
        return self.winner

    def set_inact_ranks(self):
        # Order bidders by time since last win.
        num_of_bidders = len(self.bidders)
        self.bidders = sorted(
            self.bidders, key=lambda bidder: bidder.get_time_since_last_win())
        for index, bidder in enumerate(self.bidders):
            bidder.set_inact_rank(index / num_of_bidders)

        # If they are equal, give them the same rank.
        for index, bidder in enumerate(self.bidders):
            if index != 0 and bidder.get_time_since_last_win() == self.bidders[index-1].get_time_since_last_win():
                bidder.set_inact_rank(self.bidders[index-1].get_inact_rank())

    def run_auction(self, debug=False):
        highest_bid_holder = None
        # First, set the inactivity ranks
        self.set_inact_ranks()
        # Then, set the boosted bids
        for bidder in self.bidders:
            bidder.set_boosted_bid(
                bidder.get_valuation() + (bidder.get_inact_rank() * self.inactivity_boost))
            if debug:
                print("Bidder with valuation ", str(round(bidder.get_valuation(), 2)), "and inactivity, ", str(bidder.get_time_since_last_win(
                )), "has rank ", str(round(bidder.get_inact_rank(), 2)), "and boosted bid ", str(round(bidder.get_boosted_bid(), 2)), "[boost=", str(round(self.inactivity_boost, 2)), "]")

        # Then, randomly order the bidders. Everyone participates since the reserve price is 0
        random.shuffle(self.bidders)

        # This is a shortcut. Instead of going through all the increments, we know that the winner will pay the 2nd highest bid + 0.0
        # Order bidders by their bids
        self.bidders = sorted(
            self.bidders, key=lambda bidder: bidder.get_boosted_bid(), reverse=True)
        highest_bid_holder = self.bidders[0]
        second_highest_bid = self.bidders[1].get_boosted_bid()
        # At this stage, there is only one bidder left, and they are the winner
        self.revenue = second_highest_bid + 0.01
        self.winner = highest_bid_holder

        for bidder in self.bidders:
            # Increase time waited since last win.
            bidder.increase_time_since_last_win()


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
    inactivity_boost_min_max = [0, 5]
    total_number_of_auctions = 2000
    valuations_min_max = [0, 1]
    # Holds the auction results, tuples: boosts, revenues, max_time_waited
    time_rank_per_boost_per_auction = []
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
        time_rank_per_boost_per_auction.append(
            auction.get_end_of_auction_stats())

        # Update the bandit valuations for next auction
        auction_modifier.update_bandit_valuations(auction.get_end_of_auction_stats()[
            0], auction.get_end_of_auction_stats()[1])  # We train on time_waited_rank, this  [1]

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

    return time_rank_per_boost_per_auction, auction_modifier


if __name__ == '__main__':
    results_folder_name = str(os.path.basename(__file__))
    results_folder_name = results_folder_name[:-3]
    # Evaluations are now the time_waited rankings. Higher is better, bidder that waited most received priority. (0 to 1).
    evaluations_over_time_all_sims_adaptive = []
    evaluations_over_time_all_sims_random = []
    auction_modifiers = []
    num_of_sims = 2000

    pool = Pool()  # Default number of processes will be used

    with tqdm(total=num_of_sims) as pbar:
        for results in pool.imap(run_simulation, ["adaptive"] * num_of_sims):
            evaluations_over_time_all_sims_adaptive.append(results[0])
            auction_modifiers.append(results[1])
            pbar.update()

    plot_average_bandit_valuations(auction_modifiers)

    with tqdm(total=num_of_sims) as pbar:
        for results in pool.imap(run_simulation, ["random"] * num_of_sims):
            evaluations_over_time_all_sims_random.append(results[0])
            pbar.update()

    pool.close()
    pool.join()

    # Here we plot the evaluations over time
    plot_metric_over_time(results_folder_name,
                          evaluations_over_time_all_sims_adaptive, evaluations_over_time_all_sims_random, "Time Waited Rank\n Higher better")
