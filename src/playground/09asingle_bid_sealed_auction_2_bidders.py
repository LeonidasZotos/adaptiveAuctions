"""In this file, 2 bidders participate in single-bid sealed auctions, with different loss aversions"""
import os
import numpy as np
import random
from math import exp, inf
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


def plot_average_revenue_per_reserve(results_folder_name, results, counts):
    # Counts are divided by 2, so that the size is not too big
    counts = [count/2 for count in counts]
    x = np.array([result[0] for result in results])
    y = np.array([result[1] for result in results])

    # for each x, calculate the average y value
    x_unique = np.unique(x)
    y_unique = np.array([np.mean(y[x == i]) for i in x_unique])

    # The size of each marker is proportional to the number of times that reserve price was used
    plt.scatter(x_unique, y_unique, s=counts, color='black')
    plt.xlabel("Reserve price")
    plt.ylabel("Revenue")

    if not os.path.exists(results_folder_name):
        os.makedirs(results_folder_name)
    plt.savefig(results_folder_name + '/average_revenue_per_reserve.png')
    plt.close()


def plot_revenue_over_time(results_folder_name, revenues_adaptive, revenues_random):
    import matplotlib.pyplot as plt
    import numpy as np

    # calculate the average revenue for each auction
    x = np.array([i for i in range(len(revenues_adaptive[0]))])
    y_adaptive = np.array([np.mean([revenues_adaptive[i][j] for i in range(len(revenues_adaptive))])
                           for j in range(len(revenues_adaptive[0]))])
    y_random = np.array([np.mean([revenues_random[i][j] for i in range(len(revenues_random))])
                         for j in range(len(revenues_random[0]))])

    plt.plot(x, y_adaptive, label="adaptive")
    plt.plot(x, y_random, label="random")
    plt.xlabel("Auction number")
    plt.ylabel("Revenue")
    plt.legend()

    if not os.path.exists(results_folder_name):
        os.makedirs(results_folder_name)
    plt.savefig(results_folder_name + '/revenue_over_time.png')
    plt.close()


class AuctionModifier:
    def __init__(self, reserve_min_max, num_of_auctions):
        self.num_of_auctions = num_of_auctions
        self.discretization = 13
        # possible prices are between reserve_min_max[0] and reserve_min_max[1], spaced out evenly
        self.reserve_prices = list(np.linspace(
            reserve_min_max[0], reserve_min_max[1], self.discretization))

        self.bandit_params = {}   # Bandit adaptive parameters
        self.init_bandit_params()

    def get_counts(self):
        return self.bandit_params['counts']

    def init_bandit_params(self):
        uninformed_score = 0.6
        initial_temperature = 0.1
        final_temperature = 0.01
        # calculate the decay needed to reach the final temperature after num_of_auctions auctions
        temperature_decay = (initial_temperature -
                             final_temperature) / self.num_of_auctions
        counts = [1] * len(self.reserve_prices)
        average_scores = [uninformed_score] * len(self.reserve_prices)
        boltzmann_probabilities = [0] * len(self.reserve_prices)

        for prob_index, _ in enumerate(boltzmann_probabilities):
            boltzmann_probabilities[prob_index] = round(exp(
                average_scores[prob_index]/initial_temperature), 2)
        sum_of_boltzmann_probabilities = sum(boltzmann_probabilities)
        for prob in boltzmann_probabilities:
            prob = prob/sum_of_boltzmann_probabilities

        self.bandit_params = {'possible_reserve_prices': self.reserve_prices,
                              'temperature_decay': temperature_decay,
                              'counts': counts,
                              'average_scores': average_scores,
                              'current_temperature': initial_temperature
                              }

    def generate_reserve_price(self):
        """Returns the reserve price for the next auction, using the bandit adaptive algorithm."""
        # First, reduce the temperature
        self.bandit_params['current_temperature'] = self.bandit_params['current_temperature'] - \
            self.bandit_params['temperature_decay']

        # Then, calculate the Boltzmann probabilities.
        boltzmann_probabilities = [
            0] * len(self.bandit_params['possible_reserve_prices'])

        for prob_index, _ in enumerate(boltzmann_probabilities):
            try:
                boltzmann_probabilities[prob_index] = exp(
                    self.bandit_params['average_scores'][prob_index]/self.bandit_params['current_temperature'])
            except OverflowError:
                boltzmann_probabilities[prob_index] = inf

        sum_of_boltzmann_probabilities = sum(boltzmann_probabilities)
        for prob in boltzmann_probabilities:
            prob = prob/sum_of_boltzmann_probabilities

        # Last, choose a reserve price based on the Boltzmann probabilities.
        chosen_reserve_price = random.choices(
            self.bandit_params['possible_reserve_prices'], weights=boltzmann_probabilities)

        # if there is a parameter with count 1, choose that one, so that we try it out.
        if 1 in self.bandit_params['counts']:
            chosen_reserve_price = [
                self.bandit_params['possible_reserve_prices'][self.bandit_params['counts'].index(1)]]

        return chosen_reserve_price[0]

    def update_bandit_valuations(self, reserve_price, revenue):
        # placeholder: return random between 0 and 1
        params_index = self.bandit_params['possible_reserve_prices'].index(
            reserve_price)
        self.bandit_params['counts'][params_index] += 1
        self.bandit_params['average_scores'][params_index] = (self.bandit_params['average_scores'][params_index] * (
            self.bandit_params['counts'][params_index]) + revenue) / (self.bandit_params['counts'][params_index] + 1)


class Bidder:
    def __init__(self, valuation, loss_aversion, valuations_min_max, aversions_min_max, mu_a, sigma_a):
        self.aversion_change = 0.001
        # Will be different for the competitor, as in Pardoe 2006
        self.valuation = valuation
        self.aversion = loss_aversion
        self.valuations_min_max = valuations_min_max
        self.aversions_min_max = aversions_min_max
        self.mu_a = mu_a
        self.sigma_a = sigma_a
        self.just_won_auction = False

    def win_auction(self):
        self.just_won_auction = True

    def lose_auction(self):
        self.just_won_auction = False

    def will_participate_in_auction(self, reserve_price):
        # Returns whether the bidder will participate in the auction
        return self.valuation > reserve_price

    def submit_bid(self):
        # The bidder bids more aggressively if they just lost an auction. The opposite is true if they just won an auction.
        # In any case, they bid more aggressively due to the initial loss aversion
        if self.just_won_auction:
            # If the bidder just won an auction, loss aversion is decreased.
            self.aversion = max(
                self.aversion-self.aversion_change, self.aversions_min_max[0])
        if not self.just_won_auction:
            # If the bidder just lost an auction, loss aversion is increased.
            self.aversion = min(
                self.aversion+self.aversion_change, self.aversions_min_max[1])

        return self.valuation * self.aversion


class Auction:
    def __init__(self, reserve_price, bidders):
        self.bidder1 = bidders[0]
        self.bidder2 = bidders[1]
        self.reserve_price = reserve_price
        self.revenue = 0
        self.winner = None

    def get_end_of_auction_stats(self):
        return self.reserve_price, self.revenue

    def run_auction(self):
        winning_bid = 0  # stays at 0 if no one bids above the reserve price
        auction_winner = None
        # Randomly choose who bids first

        if self.bidder1.will_participate_in_auction(self.reserve_price):
            bid1 = self.bidder1.submit_bid()
            if bid1 > winning_bid:
                winning_bid = bid1
                auction_winner = self.bidder1
        if self.bidder2.will_participate_in_auction(self.reserve_price):
            bid2 = self.bidder2.submit_bid()
            if bid2 > winning_bid:
                winning_bid = bid2
                auction_winner = self.bidder2

        # All bidders that are not the winner lose the auction
        if auction_winner == self.bidder1:
            self.bidder2.lose_auction()
        if auction_winner == self.bidder2:
            self.bidder1.lose_auction()

        if auction_winner is not None:
            auction_winner.win_auction()
            self.revenue = winning_bid
            self.winner = auction_winner
            # print("participation", winning_bid, "(reserve price: ", self.reserve_price,
            #       ", winner aversion: ", self.winner.aversion, ", winner: ", self.winner, ")")
        if auction_winner is None:
            # If no one bids above the reserve price/participates, the revenue is 0
            self.revenue = 0
            # print("no participation", "(reserve price: ", self.reserve_price, ")")


def run_simulation(reserve):
    reserve_min_max = [0, 1]
    total_number_of_auctions = 2000
    number_of_auctions_per_set_of_bidders = 20
    valuations_min_max = [0, 1]
    aversions_min_max = [1, 2.5]

    reserves_and_revenues = []  # Holds the auction results
    revenues_over_time = []  # Holds the revenue for each auction
    auction_modifier = AuctionModifier(
        reserve_min_max, total_number_of_auctions)

    # These are initialised here so that they can be used in the while loop. They change every number_of_auctions_per_set_of_bidders auctions
    mu_v, sigma_v = 0, 0
    mu_a, sigma_a = 0, 0
    v_bidder1, v_bidder2, a_bidder1, a_bidder2 = 0, 0, 0, 0
    mu_v, sigma_v = np.random.uniform(
        valuations_min_max[0], valuations_min_max[1]), 10**np.random.uniform(-2, 1)
    mu_a, sigma_a = np.random.uniform(
        aversions_min_max[0], aversions_min_max[1]), 10**np.random.uniform(-2, 1)
    bidders = []

    for auction in range(total_number_of_auctions):
        # create gaussian distribution of valuations, with mean randomly picked between 0 and 1 and variance 10^x where x is randomly picked between -2 and 1
        # create gaussian distribution of loss aversions, with mean randomly picked between 1 and 2.5 and variance 10^x where x is randomly picked between -2 and 1

        if (auction % number_of_auctions_per_set_of_bidders == 0):
            v_bidder1 = np.random.normal(mu_v, sigma_v)
            v_bidder2 = np.random.normal(mu_v, sigma_v)
            # Different loss_aversions for both bidders, not as in Pardoe 2006
            a_bidder1 = np.random.normal(mu_a, sigma_a)
            a_bidder2 = np.random.normal(mu_a, sigma_a)

            # Redraw if valuation is outside of the accepted range.
            while v_bidder1 < valuations_min_max[0] or v_bidder1 > valuations_min_max[1]:
                v_bidder1 = np.random.normal(mu_v, sigma_v)
            while v_bidder2 < valuations_min_max[0] or v_bidder2 > valuations_min_max[1]:
                v_bidder2 = np.random.normal(mu_v, sigma_v)
            while a_bidder1 < aversions_min_max[0] or a_bidder1 > aversions_min_max[1]:
                a_bidder1 = np.random.normal(mu_a, sigma_a)
            while a_bidder2 < aversions_min_max[0] or a_bidder2 > aversions_min_max[1]:
                a_bidder2 = np.random.normal(mu_a, sigma_a)
            # Create the two bidders that will participate in the auction
            bidders = [Bidder(v_bidder1, a_bidder1, valuations_min_max, aversions_min_max, mu_a, sigma_a), Bidder(
                v_bidder2, a_bidder2, valuations_min_max, aversions_min_max, mu_a, sigma_a)]

        reserve_price = 0
        if reserve == 'adaptive':
            reserve_price = auction_modifier.generate_reserve_price()
        elif reserve == 'random':
            reserve_price = random.choices(list(np.linspace(
                reserve_min_max[0], reserve_min_max[1], 13)))[0]  # randomly pick a reserve price
        auction = Auction(reserve_price, bidders)
        auction.run_auction()
        reserves_and_revenues.append(auction.get_end_of_auction_stats())
        revenues_over_time.append(auction.get_end_of_auction_stats()[1])

        # Update the reserve price for the next auction
        auction_modifier.update_bandit_valuations(auction.get_end_of_auction_stats()[
            0], auction.get_end_of_auction_stats()[1])

    return reserves_and_revenues, revenues_over_time, auction_modifier


if __name__ == '__main__':
    results_folder_name = str(os.path.basename(__file__))
    results_folder_name = results_folder_name[:-3]
    revenues_over_time_all_sims_adaptive = []
    revenues_over_time_all_sims_random = []
    last_reserves_and_revenues = []
    last_auction_modifier = None
    num_of_sims = 2000

    pool = Pool()  # Default number of processes will be used

    with tqdm(total=num_of_sims) as pbar:
        for results in pool.imap(run_simulation, ["adaptive"] * num_of_sims):
            last_reserves_and_revenues = results[0]
            last_auction_modifier = results[2]
            revenues_over_time_all_sims_adaptive.append(results[1])
            pbar.update()

    with tqdm(total=num_of_sims) as pbar:
        for results in pool.imap(run_simulation, ["random"] * num_of_sims):
            last_reserves_and_revenues = results[0]
            last_auction_modifier = results[2]
            revenues_over_time_all_sims_random.append(results[1])
            pbar.update()

    pool.close()
    pool.join()
    plot_average_revenue_per_reserve(results_folder_name,
                                     last_reserves_and_revenues, last_auction_modifier.get_counts())
    plot_revenue_over_time(results_folder_name,
                           revenues_over_time_all_sims_adaptive, revenues_over_time_all_sims_random)
