"""In this file, a minimal replication of the Pardoe 2006 paper is attempted, excluding the meta-learning"""
import numpy as np
import random
from math import exp
from scipy.integrate import quad
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


def plot_average_revenue_per_reserve(results, counts):
    # Counts are divided by 2, so that the size is not too big
    counts = [count/2 for count in counts]

    x = np.array([result[0] for result in results])
    y = np.array([result[1] for result in results])

    # for each x, calculate the average y value
    x_unique = np.unique(x)
    y_unique = np.array([np.mean(y[x == i]) for i in x_unique])
    # the size of each marker is proportional to the number of times that reserve price was used
    plt.scatter(x_unique, y_unique, s=counts, color='black')
    plt.xlabel("Reserve price")
    plt.ylabel("Revenue")
    plt.show()


def plot_revenue_over_time(revenues):
    import matplotlib.pyplot as plt
    import numpy as np

    # calculate the average revenue for each auction
    x = np.array([i for i in range(len(revenues[0]))])
    y = np.array([np.mean([revenues[i][j] for i in range(len(revenues))])
                  for j in range(len(revenues[0]))])

    plt.plot(x, y, 'o', color='black')
    plt.xlabel("Auction number")
    plt.ylabel("Revenue")
    plt.show()


def uniform_distribution_function(x, a, b):
    if x < a or x > b:
        return 0
    return 1 / (b - a)


class AuctionModifier:
    def __init__(self, reserve_min_max, num_of_auctions, discretization=13, ):
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
            except:
                print(
                    "ERROR: Error occured when trying to calculate the Boltzmann probabilities")
                print("attempted to calc: exp(",
                      self.bandit_params['average_scores'][prob_index], "/", self.bandit_params['current_temperature'])

        sum_of_boltzmann_probabilities = sum(boltzmann_probabilities)
        for prob in boltzmann_probabilities:
            prob = prob/sum_of_boltzmann_probabilities

        # Last, choose a reserve price based on the Boltzmann probabilities.
        chosen_reserve_price = random.choices(
            self.bandit_params['possible_reserve_prices'], weights=boltzmann_probabilities)

        return chosen_reserve_price[0]

    def update_bandit_valuations(self, reserve_price, revenue):
        # placeholder: return random between 0 and 1
        params_index = self.bandit_params['possible_reserve_prices'].index(
            reserve_price)
        self.bandit_params['counts'][params_index] += 1
        self.bandit_params['average_scores'][params_index] = (self.bandit_params['average_scores'][params_index] * (
            self.bandit_params['counts'][params_index]) + revenue) / (self.bandit_params['counts'][params_index] + 1)


class Bidder:
    def __init__(self, valuation, loss_aversion, valuations_min_max):
        # Will be different for the competitor, as in Pardoe 2006
        self.valuation = valuation
        # Will be the same for the competitor, as in Pardoe 2006
        self.aversion = loss_aversion
        self.valuations_min_max = valuations_min_max

    def integral_equilibrium_function(self, v1):
        return (self.valuation - (self.aversion * v1)) * uniform_distribution_function(v1, self.valuations_min_max[0], self.valuations_min_max[1])

    def equilibrium_result_is_positive(self, reserve_price):
        # Returns whether the equilibrium function is positive
        return quad(self.integral_equilibrium_function, reserve_price, self.valuation)[0] > 0

    def will_participate_in_auction_as_first_bidder(self, reserve_price):
        # Returns whether the bidder will participate in the auction
        return self.valuation > reserve_price

    def will_participate_in_auction_as_second_bidder(self, reserve_price):
        # Returns whether the bidder will participate in the auction
        return self.equilibrium_result_is_positive(reserve_price)

    def bid_further(self, current_price):
        # This is only executed if the bidder is participating in the auction. Returns True if the bidder wants to stay in the auction, False if not.
        if current_price < self.valuation * self.aversion:
            return True
        return False


class Auction:
    def __init__(self, reserve_price, bidders, increment):
        self.bidder1 = bidders[0]
        self.bidder2 = bidders[1]
        self.increment = increment
        self.reserve_price = reserve_price
        self.revenue = 0
        self.winner = None

    def get_end_of_auction_stats(self):
        # print("At a reserve price of", round(self.reserve_price, 3),
        #       "the revenue is", round(self.revenue, 3))
        return self.reserve_price, self.revenue

    def run_auction(self):
        current_bid = 0
        highest_bid_holder = None
        # Randomly choose who bids first
        first_bidder = np.random.choice([self.bidder1, self.bidder2])
        second_bidder = self.bidder1 if first_bidder == self.bidder2 else self.bidder2
        # If the 1st bidder participates in the auction:
        if first_bidder.will_participate_in_auction_as_first_bidder(self.reserve_price):
            current_bid = self.reserve_price
            highest_bid_holder = first_bidder
            if second_bidder.will_participate_in_auction_as_second_bidder(self.reserve_price):
                current_bid += self.increment
                highest_bid_holder = second_bidder
                # Run auction with both bidders
                while True:
                    if first_bidder.bid_further(current_bid):
                        current_bid += self.increment
                        highest_bid_holder = first_bidder
                    else:
                        self.revenue = current_bid
                        self.winner = highest_bid_holder
                        break

                    if second_bidder.bid_further(current_bid):
                        current_bid += self.increment
                        highest_bid_holder = second_bidder
                    else:
                        self.revenue = current_bid
                        self.winner = highest_bid_holder
                        break

            else:
                # The 2nd bidder does not participate, so the auction is over
                self.revenue = current_bid
                self.winner = highest_bid_holder

        else:
            # If the 1st bidder does not participate in the auction:
            if second_bidder.will_participate_in_auction_as_first_bidder(self.reserve_price):
                current_bid = self.reserve_price
                highest_bid_holder = second_bidder
                # Here, the 1st bidder does not participate, and only second_bidder participates, so the auction is over
                self.revenue = current_bid
            else:
                # Here, neither bidder participates, so the auction is over. Returns 0 as revenue and None as winner
                self.revenue = current_bid
                self.winner = highest_bid_holder


def run_simulation(id):
    sim_id = id
    increments = 0.001  # This is not defined in Pardoe 2006
    reserve_min_max = [0, 1]
    number_of_auctions = 6000
    valuations_min_max = [0, 1]
    aversions_min_max = [1, 2.5]

    reserves_and_revenues = []  # Holds the auction results
    revenues_over_time = []  # Holds the revenue for each auction
    auction_modifier = AuctionModifier(reserve_min_max, number_of_auctions)

    for i in range(number_of_auctions):
        # create gaussian distribution of valuations, with mean randomly picked between 0 and 1 and variance 10^x where x is randomly picked between -2 and 1
        # create gaussian distribution of loss aversions, with mean randomly picked between 1 and 2.5 and variance 10^x where x is randomly picked between -2 and 1
        mu_v, sigma_v = np.random.uniform(
            valuations_min_max[0], valuations_min_max[1]), 10**np.random.uniform(-2, 1)
        mu_a, sigma_a = np.random.uniform(
            aversions_min_max[0], aversions_min_max[1]), 10**np.random.uniform(-2, 1)

        v_bidder1 = np.random.normal(mu_v, sigma_v)
        v_bidder2 = np.random.normal(mu_v, sigma_v)
        # Same loss_aversion for both bidders, as in Pardoe 2006
        a_bidder = np.random.normal(mu_a, sigma_a)

        # Redraw if valuation is outside of the accepted range.
        while v_bidder1 < valuations_min_max[0] or v_bidder1 > valuations_min_max[1]:
            v_bidder1 = np.random.normal(mu_v, sigma_v)
        while v_bidder2 < valuations_min_max[0] or v_bidder2 > valuations_min_max[1]:
            v_bidder2 = np.random.normal(mu_v, sigma_v)
        while a_bidder < aversions_min_max[0] or a_bidder > aversions_min_max[1]:
            a_bidder = np.random.normal(mu_a, sigma_a)

        # Create the two bidders that will participate in the auction
        bidders = [Bidder(v_bidder1, a_bidder, valuations_min_max), Bidder(
            v_bidder2, a_bidder, valuations_min_max)]

        auction = Auction(
            auction_modifier.generate_reserve_price(), bidders, increments)
        auction.run_auction()
        reserves_and_revenues.append(auction.get_end_of_auction_stats())
        revenues_over_time.append(auction.get_end_of_auction_stats()[1])

        # Update the reserve price for the next auction
        auction_modifier.update_bandit_valuations(auction.get_end_of_auction_stats()[
            0], auction.get_end_of_auction_stats()[1])

    return reserves_and_revenues, revenues_over_time, auction_modifier


if __name__ == '__main__':

    revenues_over_time_all_sims = []
    last_reserves_and_revenues = []
    last_auction_modifier = None

    pool = Pool()  # Default number of processes will be used

    with tqdm(total=500) as pbar:
        for results in pool.imap(run_simulation, range(500)):
            last_reserves_and_revenues = results[0]
            last_auction_modifier = results[2]
            revenues_over_time_all_sims.append(results[1])
            pbar.update()

    pool.close()
    pool.join()

    plot_average_revenue_per_reserve(
        last_reserves_and_revenues, last_auction_modifier.get_counts())

    plot_revenue_over_time(revenues_over_time_all_sims)
