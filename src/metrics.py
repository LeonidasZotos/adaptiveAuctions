"""A class to keep track of the metrics of the simulation and create relevant graphs."""
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from math import nan
import src.utils as utils

NUM_OF_ADAPT_PARAMS = 1
WARMUP_EPOCHS = 0


class MasterKeeper:
    """
    The MasterKeeper class is responsible for keeping track of all the metrics for all simulations.
    Attributes:
        all_simulations_results (list): A list of dictionaries, each dictionary containing the satisfaction scores of all cars
            that completed their trip in an epoch, for a single simulation.
        total_population_per_intersection_all_sims (np.array): A 3d array of the population of each intersection for each simulation. The first index is the x coordinate,
            the second is the y coordinate, the third is the simulation. The value is the population.
        total_throughput_history_per_intersection_all_sims (np.array): A 3d array of the throughput history of each intersection. The first index is the x coordinate,
            the second is the y coordinate, the third is the epoch. The value is the throughput.
        count_of_reward_measurements_per_intersection (np.array): A 3d array of the number of measurements of each intersection. The first index is the x coordinate,
            the second is the y coordinate, the third is the epoch. The value is the number of measurements.
        total_reward_history_per_intersection_all_sims (np.array): A 3d array of the reward history of each intersection. The first index is the x coordinate,
            the second is the y coordinate, the third is the epoch. The value is the reward.
        count_of_max_time_waited_measurements_per_intersection (np.array): A 3d array of the number of measurements of each intersection. The first index is the x coordinate,
            the second is the y coordinate, the third is the epoch. The value is the number of measurements.
        max_time_waited_history_per_intersection_all_sims (np.array): A 3d array of the max time waited history of each intersection. The first index is the x coordinate,
            the second is the y coordinate, the third is the epoch. The value is the max time waited.
        auction_parameters_space (np.array): A 2d array of the auction parameters space. The first index is the delay boost, the second is the queue length boost.
        sum_auction_parameters_valuations_per_intersection (np.array): A 3d array of the sum of the valuations of the auction parameters of each intersection.
            The first index is the x coordinate, the second is the y coordinate, the third is the parameter space. The value is the sum of the valuations.
        sum_auction_parameters_counts_per_intersection (np.array): A 3d array of the sum of the counts of the trials for each parameter comb. of each intersection.
    Functions:
        store_simulation_results(sim_metrics_keeper): Stores the results of a single simulation
        produce_results(): Produces all the evaluation results of all simulations
        produce_general_metrics(): Produces the general metrics of all simulations
        plot_satisfaction_scores_overall_average(): Creates a graph of the average satisfaction score per epoch, with error bars, averaged over all simulations
        plot_satisfaction_scores_by_bidding_type(with_std=False, export_results=True, filter_outliers=True): Creates a
            graph of the average satisfaction score per epoch, with error bars, averaged over all simulations,
            for each bidding type, represented by a different color.
        histogram_satisfaction_scores_by_bidding_type(export_results=True, filter_outliers=True): Creates a histogram of all satisfaction scores,
            over all simulations, per bidding type
        plot_throughput_heatmap_average(export_results=True): Creates a heatmap of the
            average tƒhroughput per intersection, over all simulations
        plot_throughput_per_intersection_history(export_results=True): Creates a plot with subplots for each intersection.
            Each subplot is a graph of the throughput history of that intersection. In total there are as many subplots as intersections
        plot_reward_per_intersection_history(export_results=True): Creates a plot with subplots for each intersection.
            Each subplot is a graph of the reward history of that intersection. In total there are as many subplots as intersections
        plot_adaptive_auction_parameters_valuations_per_intersection(export_results=True): Creates a plot with subplots for each intersection.
            Each subplot is a graph of the valuations of the auction parameters of that intersection.
    """

    ### General Functions ###
    def __init__(self, args):
        """ Initialize the MetricsKeeper object
        Args:
            args (argparse.Namespace): Arguments parsed from the command line
        """
        self.args = args

        self.export_location = self.args.results_folder + "/exported_data"
        if not os.path.exists(self.export_location):
            os.makedirs(self.export_location)

        # A dictionary where general metrics are kept:
        self.general_metrics = {}

        # Number of Gridlocked simulations
        self.num_of_gridlocks = 0

        ####### Trip Satisfaction Metrics #######
        # The satisfaction scores history of all simulations
        self.all_simulations_satisfaction_scores = []

        ####### Congestion Metrics #######
        # Total throughput per intersection
        self.total_population_per_intersection_all_sims = []

        # Total throughput history per intersection
        self.total_throughput_history_per_intersection_all_sims = []

        ####### Time Waited Metrics #######
        # Number of average_time_waited measurements per intersection, used to calculate the average
        self.count_of_average_time_waited_measurements_per_intersection = np.zeros(
            (self.args.grid_size, self.args.grid_size, self.args.num_of_epochs))

        # Average time waited history per intersection
        self.average_time_waited_history_per_intersection_all_sims = []

        # Number of max_time_waited measurements per intersection, used to calculate the average
        self.count_of_max_time_waited_measurements_per_intersection = np.zeros(
            (self.args.grid_size, self.args.grid_size, self.args.num_of_epochs))

        # Max time waited history per intersection
        self.max_time_waited_history_per_intersection_all_sims = []

        # Gini history per intersection
        self.gini_time_waited_history_per_intersection_all_sims = []

        # Number of gini measurements per intersection, used to calculate the average
        self.count_of_gini_time_waited_measurements_per_intersection = np.zeros(
            (self.args.grid_size, self.args.grid_size, self.args.num_of_epochs))

        ####### Winner Worthiness/Auction Reward Metrics #######
        # Number of reward measurements per intersection, used to calculate the average
        self.count_of_reward_measurements_per_intersection = np.zeros(
            (self.args.grid_size, self.args.grid_size, self.args.num_of_epochs))

        # Total reward history per intersection
        self.reward_history_per_simulation_all_sims = []

        # Auction parameter space
        self.auction_parameters_space = []

        # Sum of auction parameters valuations per intersection
        self.sum_auction_parameters_valuations_per_intersection = np.zeros(
            (self.args.grid_size, self.args.grid_size, pow(self.args.adaptive_auction_discretization, NUM_OF_ADAPT_PARAMS)))

        # Count of auction parameters trials per intersection
        self.sum_auction_parameters_counts_per_intersection = np.zeros(
            (self.args.grid_size, self.args.grid_size, pow(self.args.adaptive_auction_discretization, NUM_OF_ADAPT_PARAMS)))

        # Winner inact and bid ranks per intersection. The length of the third dimension is usually num_of_simulations, but if an intersection never held an auction, it will be smaller.
        self.all_sims_winners_inact_ranks_means = [
            [[] for _ in range(args.grid_size)] for _ in range(args.grid_size)]
        self.all_sims_winners_bid_ranks_means = [
            [[] for _ in range(args.grid_size)] for _ in range(args.grid_size)]

        ####### Misc. #######
        self.all_sims_broke_agents_history = np.zeros(self.args.num_of_epochs)

    def store_simulation_results(self, sim_metrics_keeper):
        """Prepares the metrics keeper for a new simulation, by clearing the results of the current simulation
        Args:
            sim_metrics_keeper (SimulationMetrics): The metrics keeper of the current simulation
        """

        if sim_metrics_keeper.check_if_gridlocked() == True:
            # If the simulation was gridlocked, take note of that and do not store the results.
            self.num_of_gridlocks += 1
            return
        ### Satisfaction Metric ###
        self.all_simulations_satisfaction_scores.append(
            sim_metrics_keeper.current_sim_satisfaction_scores)

        ### Congestion Metric ###
        self.total_population_per_intersection_all_sims.append(
            sim_metrics_keeper.total_population_per_intersection)

        self.total_throughput_history_per_intersection_all_sims.append(
            sim_metrics_keeper.throughput_history_per_intersection)

        ### Time Waited Metrics ###
        ## Average ##
        self.count_of_average_time_waited_measurements_per_intersection += np.where(
            np.isnan(sim_metrics_keeper.average_time_waited_history_per_intersection), 0, 1)
        self.average_time_waited_history_per_intersection_all_sims.append(np.nan_to_num(
            sim_metrics_keeper.average_time_waited_history_per_intersection))
        ## Max ##
        self.count_of_max_time_waited_measurements_per_intersection += np.where(
            np.isnan(sim_metrics_keeper.max_time_waited_history_per_intersection), 0, 1)
        self.max_time_waited_history_per_intersection_all_sims.append(np.nan_to_num(
            sim_metrics_keeper.max_time_waited_history_per_intersection))
        ## Gini Time Waited ##
        self.count_of_gini_time_waited_measurements_per_intersection += np.where(
            np.isnan(sim_metrics_keeper.gini_time_waited_history_per_intersection), 0, 1)
        self.gini_time_waited_history_per_intersection_all_sims.append(np.nan_to_num(
            sim_metrics_keeper.gini_time_waited_history_per_intersection))

        ### Auction Metrics ###
        # For each measurement that is not nan, we add 1 to the count of measurements, so that we can later calculate the average
        self.count_of_reward_measurements_per_intersection += np.where(
            np.isnan(sim_metrics_keeper.reward_history_per_intersection), 0, 1)
        self.reward_history_per_simulation_all_sims.append(
            np.nan_to_num(sim_metrics_keeper.reward_history_per_intersection))

        # Retrieve the parameter space and the valuations per intersection. The parameters space is the same for all intersections
        self.auction_parameters_space = sim_metrics_keeper.auction_parameters_space

        self.sum_auction_parameters_valuations_per_intersection += sim_metrics_keeper.auction_parameters_valuations_per_intersection
        self.sum_auction_parameters_counts_per_intersection += sim_metrics_keeper.auction_parameters_counts_per_intersection

        # Store the mean and sd bid and inact rank from the simulation
        means = sim_metrics_keeper.winners_inact_ranks_per_intersection_means
        valid_indices = ~np.isnan(means)
        grid_size = self.args.grid_size
        for i in range(grid_size):
            for j in range(grid_size):
                if valid_indices[i, j]:
                    self.all_sims_winners_inact_ranks_means[i][j].append(
                        means[i, j])
        means = sim_metrics_keeper.winners_bid_ranks_per_intersection_means
        valid_indices = ~np.isnan(means)
        grid_size = self.args.grid_size
        for i in range(grid_size):
            for j in range(grid_size):
                if valid_indices[i, j]:
                    self.all_sims_winners_bid_ranks_means[i][j].append(
                        means[i, j])
        ### Misc. ###
        self.all_sims_broke_agents_history += sim_metrics_keeper.broke_history

    def produce_results(self):
        """Produces all the evaluation results of all simulations"""

        # Create a .txt file with the arguments used for the simulation
        with open(self.args.results_folder + '/configuration.txt', 'w') as f:
            for arg in vars(self.args):
                f.write(arg + ': ' + str(getattr(self.args, arg)) + '\n')

        ### General, non-plot, Metrics ###
        self.produce_general_metrics()

        if self.args.sweep_mode:
            # If this is activated, don't produce any plots
            return
        
        ### Satisfaction Metrics ###
        # Create a graph of all satisfaction scores, over all simulations
        self.plot_satisfaction_scores_overall_average()
        # Create a graph of all satisfaction scores, per bidding type, over all simulations
        self.plot_satisfaction_scores_by_bidding_type()
        # Create a histogram of all satisfaction scores, over all simulations, per bidding type
        self.plot_histogram_satisfaction_scores_by_bidding_type()

        ### Congestion Metrics ###
        # Create a heatmap of the average throughput per intersection, over all simulations
        self.plot_congestion_heatmap_average()
        # Create a graph with graphs of the average throughput per intersection, over all simulations
        self.plot_throughput_per_intersection_history()

        ### Time Waited Metrics ###
        # Create a graph with graphs of the average average time waited per intersection, over all simulations
        self.plot_average_time_waited_per_intersection_history()
        # Create a graph with graphs of the average max time waited per intersection, over all simulations
        self.plot_max_time_waited_per_intersection_history()
        # Create a graph with graphs of the average gini time waited per intersection, over all simulations
        self.plot_gini_time_waited_per_intersection_history()

        ### Auction Metrics ###
        # Create a graph with graphs of the average reward per intersection, over all simulations
        self.plot_reward_per_intersection_history()
        # Create a graph with the valuations of the auction parameters of each intersection
        self.plot_adaptive_auction_parameters_valuations_per_intersection()
        # Create a barplot with the mean bid and inact rank of the winner per intersection
        self.plot_mean_bid_and_inact_rank_per_intersection()

        ### Misc. ###
        self.plot_broke_agents_percentage_history()

    def produce_general_metrics(self):
        """Produces the general metrics of all simulations"""
        # Congestion Metrics
        self.calc_central_congestion()
        # Average number of trips completed per simulation
        self.calc_average_num_of_trips_completed()

        # Time Waited Metrics
        self.calc_time_waited_general_metrics()
        self.calc_time_waited_gini_metric()

        # Satisfaction Metrics:
        self.calc_average_trip_satisfaction()
        self.calc_satisfaction_gini_metric()

        # Export the calculated metrics into a .txt file
        with open(self.args.results_folder + '/general_metrics.txt', 'w') as f:
            for metric in self.general_metrics:
                f.write(
                    metric + ': \n' + str(self.general_metrics[metric]) + '\n=====================\n')
                
        # Print some overall metrics from the dictionary:
        print("End of Simulation Main Metrics:")
        print("Average Time waited: " + str(self.general_metrics['grid_average_time_waited']))
        print("Max Time waited: " + str(self.general_metrics['grid_max_time_waited']))
        print("Average Trip Satisfaction: " + str(self.general_metrics['Average Trip Satisfaction']))
        print("----------------------------------")

    ### Trip Satisfaction Metric ###
    def calc_satisfaction_gini_metric(self):
        def remove_car_copies_from_dict(dict):
            """Removes the car copies from the dictionary, so that it only contains the satisfaction scores"""
            return [score for (_, score) in dict]

        def calc_gini(x):
            """Calculates the GINI coeffecient of a list of numbers.
            Source: https://www.statology.org/gini-coefficient-python/
            """
            x = np.array(x)
            total = 0
            for i, xi in enumerate(x[:-1], 1):
                total += np.sum(np.abs(xi - x[i:]))
                with np.errstate(divide='ignore', invalid='ignore'):
                    gini_result = total / (len(x)**2 * np.mean(x))
            return gini_result

        gini_per_simulation = []
        for sim in self.all_simulations_satisfaction_scores:
            sim_satisfaction_scores = []  # That is regardless of epoch
            for epoch in sim:
                if sim[epoch] != None:
                    sim_satisfaction_scores.append(
                        remove_car_copies_from_dict(sim[epoch]))
            sim_satisfaction_scores_flat = [
                item for sublist in sim_satisfaction_scores for item in sublist]
            gini_per_simulation.append(calc_gini(sim_satisfaction_scores_flat))

        mean_gini_text = str(round(np.mean(gini_per_simulation), 3))
        std_gini_text = str(round(np.std(gini_per_simulation), 3))

        self.general_metrics['satisfacation_avg_gini'] = str(
            "Mean: " + mean_gini_text + " | SD: " + std_gini_text + " | Description: The Gini coefficient of all satisfaction scores of the simulation. Averaged over sims")

    def plot_satisfaction_scores_overall_average(self, export_results=True):
        """Creates a graph of the average satisfaction score per epoch, with error bars, averaged over all simulations.
        Args:
            export_results (bool): Whether to export the results to a .npy file
        """

        def remove_car_copies_from_dict(dict):
            """Removes the car copies from the dictionary, so that it only contains the satisfaction scores"""
            return [score for (_, score) in dict]

        all_results_dict = {}
        # First, combine all dictionaries into one dictionary
        for result_dict in self.all_simulations_satisfaction_scores:
            for epoch in result_dict:
                if epoch in all_results_dict:
                    all_results_dict[epoch] += remove_car_copies_from_dict(
                        result_dict[epoch])
                else:
                    all_results_dict[epoch] = remove_car_copies_from_dict(
                        result_dict[epoch])

        # Create a list of all epochs in which cars completed their trip
        epochs = []
        for epoch in all_results_dict:
            if all_results_dict[epoch] != None:
                epochs.append(epoch)

        # Create a list of all average satisfaction scores
        average_satisfaction_scores = []
        for epoch in epochs:
            average_satisfaction_scores.append(
                sum(all_results_dict[epoch]) / len(all_results_dict[epoch]))

        # Create a list of all standard deviations of satisfaction scores
        standard_deviations = []
        for epoch in epochs:
            standard_deviations.append(
                np.std(all_results_dict[epoch]))

        # Remove the first 30 epochs, which are the warm-up epochs
        epochs = epochs[WARMUP_EPOCHS:]
        average_satisfaction_scores = average_satisfaction_scores[WARMUP_EPOCHS:]
        standard_deviations = standard_deviations[WARMUP_EPOCHS:]

        # # Plot the average satisfaction score per epoch, with error bars
        # plt.errorbar(epochs, average_satisfaction_scores,
        #              yerr=standard_deviations, fmt='o', ls='none')

        # Plot the average satisfaction score per epoch, without error bars, using a yscale of 0-1
        plt.plot(epochs, average_satisfaction_scores,
                 'o', ls='none', markersize=1.5)
        plt.xlabel('Epoch')
        plt.ylabel('Average Satisfaction Score \n (the higher, the better)')
        plt.title('Average Satisfaction Score per Epoch')
        plt.savefig(self.args.results_folder +
                    '/average_satisfaction_score.png')
        plt.clf()
        if export_results == True:
            np.save(self.export_location + "/average_satisfaction_score.npy",
                    average_satisfaction_scores)

    def plot_satisfaction_scores_by_bidding_type(self, with_std=False, filter_outliers=False):
        """Creates a graph of the average satisfaction score per epoch, with error bars, averaged over all simulations,
            for each bidding type, represented by a different color.
            'ohhh almost 200 lines of code, that's a lot of code for just one function (but here we are)'
        Args:
            with_std (bool): Whether to include the standard deviation in the plot
            filter_outliers (bool): Whether to filter out outliers from the results
        """

        homogeneous_bidding_results = {}
        heterogeneous_bidding_results = {}
        random_bidding_results = {}
        free_rider_bidding_results = {}
        RL_bidding_results = {}
        epochs = []  # A list of all epochs in which cars completed their trip

        for result_dict in self.all_simulations_satisfaction_scores:
            for epoch in result_dict:
                for (car_copy, score) in result_dict[epoch]:
                    bidding_type = car_copy.bidding_type
                    # Homogeneous bidding
                    if bidding_type == 'homogeneous':
                        epochs.append(epoch)
                        if epoch in homogeneous_bidding_results:
                            homogeneous_bidding_results[epoch].append(score)
                        else:
                            homogeneous_bidding_results[epoch] = [score]
                    # Heterogeneous bidding
                    elif bidding_type == 'heterogeneous':
                        epochs.append(epoch)
                        if epoch in heterogeneous_bidding_results:
                            heterogeneous_bidding_results[epoch].append(score)
                        else:
                            heterogeneous_bidding_results[epoch] = [score]
                    # Random bidding
                    elif bidding_type == 'random':
                        epochs.append(epoch)
                        if epoch in random_bidding_results:
                            random_bidding_results[epoch].append(score)
                        else:
                            random_bidding_results[epoch] = [score]
                    # Free-rider bidding
                    elif bidding_type == 'free-rider':
                        epochs.append(epoch)
                        if epoch in free_rider_bidding_results:
                            free_rider_bidding_results[epoch].append(score)
                        else:
                            free_rider_bidding_results[epoch] = [score]
                    # RL bidding
                    elif bidding_type == 'RL':
                        epochs.append(epoch)
                        if epoch in RL_bidding_results:
                            RL_bidding_results[epoch].append(score)
                        else:
                            RL_bidding_results[epoch] = [score]

        # Remove duplicate epochs. Epochs stores the epochs in which any car completed a trip.
        epochs = list(set(epochs))

        # Remove outliers if necessary:
        if filter_outliers == True:
            for epoch in epochs:
                # Homogeneous bidding:
                if epoch in homogeneous_bidding_results:
                    homogeneous_bidding_results[epoch] = utils.remove_outliers(
                        homogeneous_bidding_results[epoch])
                # Heterogeneous bidding
                if epoch in heterogeneous_bidding_results:
                    heterogeneous_bidding_results[epoch] = utils.remove_outliers(
                        heterogeneous_bidding_results[epoch])
                # Random bidding:
                if epoch in random_bidding_results:
                    random_bidding_results[epoch] = utils.remove_outliers(
                        random_bidding_results[epoch])
                # Free-Rider bidding:
                if epoch in free_rider_bidding_results:
                    free_rider_bidding_results[epoch] = utils.remove_outliers(
                        free_rider_bidding_results[epoch])
                # RL bidding:
                if epoch in RL_bidding_results:
                    RL_bidding_results[epoch] = utils.remove_outliers(
                        RL_bidding_results[epoch])

        # Create a list of all average satisfaction scores
        homogeneous_bidding_average_satisfaction_scores = []
        heterogeneous_bidding_average_satisfaction_scores = []
        random_bidding_average_satisfaction_scores = []
        free_rider_bidding_average_satisfaction_scores = []
        RL_bidder_average_satisfaction_scores = []

        for epoch in epochs:
            # Homogeneous bidding:
            if epoch in homogeneous_bidding_results:
                homogeneous_bidding_average_satisfaction_scores.append(
                    sum(homogeneous_bidding_results[epoch]) / len(homogeneous_bidding_results[epoch]))
            else:
                homogeneous_bidding_average_satisfaction_scores.append(nan)
            # Heterogeneous bidding:
            if epoch in heterogeneous_bidding_results:
                heterogeneous_bidding_average_satisfaction_scores.append(
                    sum(heterogeneous_bidding_results[epoch]) / len(heterogeneous_bidding_results[epoch]))
            else:
                heterogeneous_bidding_average_satisfaction_scores.append(nan)
            # Random bidding:
            if epoch in random_bidding_results:
                random_bidding_average_satisfaction_scores.append(
                    sum(random_bidding_results[epoch]) / len(random_bidding_results[epoch]))
            else:
                random_bidding_average_satisfaction_scores.append(nan)
            # Free-Rider bidding:
            if epoch in free_rider_bidding_results:
                free_rider_bidding_average_satisfaction_scores.append(
                    sum(free_rider_bidding_results[epoch]) / len(free_rider_bidding_results[epoch]))
            else:
                free_rider_bidding_average_satisfaction_scores.append(nan)
            # RL bidding:
            if epoch in RL_bidding_results:
                RL_bidder_average_satisfaction_scores.append(
                    sum(RL_bidding_results[epoch]) / len(RL_bidding_results[epoch]))
            else:
                RL_bidder_average_satisfaction_scores.append(nan)

        # Create a list of all standard deviations of satisfaction scores
        homogeneous_bidding_sd = []
        heterogeneous_bidding_sd = []
        random_bidding_sd = []
        free_rider_bidding_sd = []
        RL_bidder_sd = []
        for epoch in epochs:
            # Homogeneous bidding:
            if epoch in homogeneous_bidding_results:
                homogeneous_bidding_sd.append(
                    np.std(homogeneous_bidding_results[epoch]))
            else:
                homogeneous_bidding_sd.append(nan)
            # Heterogeneous bidding:
            if epoch in heterogeneous_bidding_results:
                heterogeneous_bidding_sd.append(
                    np.std(heterogeneous_bidding_results[epoch]))
            else:
                heterogeneous_bidding_sd.append(nan)
            # Random bidding:
            if epoch in random_bidding_results:
                random_bidding_sd.append(
                    np.std(random_bidding_results[epoch]))
            else:
                random_bidding_sd.append(nan)
            # Free-Rider bidding:
            if epoch in free_rider_bidding_results:
                free_rider_bidding_sd.append(
                    np.std(free_rider_bidding_results[epoch]))
            else:
                free_rider_bidding_sd.append(nan)
            # RL bidding:
            if epoch in RL_bidding_results:
                RL_bidder_sd.append(
                    np.std(RL_bidding_results[epoch]))
            else:
                RL_bidder_sd.append(nan)

        # Remove the first WARMUP_EPOCHS
        epochs = epochs[WARMUP_EPOCHS:]
        homogeneous_bidding_average_satisfaction_scores = homogeneous_bidding_average_satisfaction_scores[
            WARMUP_EPOCHS:]
        homogeneous_bidding_sd = homogeneous_bidding_sd[WARMUP_EPOCHS:]

        heterogeneous_bidding_average_satisfaction_scores = heterogeneous_bidding_average_satisfaction_scores[
            WARMUP_EPOCHS:]
        heterogeneous_bidding_sd = heterogeneous_bidding_sd[WARMUP_EPOCHS:]

        random_bidding_average_satisfaction_scores = random_bidding_average_satisfaction_scores[
            WARMUP_EPOCHS:]
        random_bidding_sd = random_bidding_sd[WARMUP_EPOCHS:]
        free_rider_bidding_average_satisfaction_scores = free_rider_bidding_average_satisfaction_scores[
            WARMUP_EPOCHS:]
        free_rider_bidding_sd = free_rider_bidding_sd[WARMUP_EPOCHS:]
        RL_bidder_average_satisfaction_scores = RL_bidder_average_satisfaction_scores[
            WARMUP_EPOCHS:]
        RL_bidder_sd = RL_bidder_sd[WARMUP_EPOCHS:]

        if with_std == True:
            # Plot the average satisfaction score per epoch, per bidding type & with error bars
            if len(homogeneous_bidding_results) > 0:
                plt.errorbar(epochs, homogeneous_bidding_average_satisfaction_scores,
                             yerr=homogeneous_bidding_sd, fmt='o', label='Homogeneous bidding')
            if len(heterogeneous_bidding_results) > 0:
                plt.errorbar(epochs, heterogeneous_bidding_average_satisfaction_scores,
                             yerr=heterogeneous_bidding_sd, fmt='o', label='Heterogeneous bidding')
            if len(random_bidding_results) > 0:
                plt.errorbar(epochs, random_bidding_average_satisfaction_scores,
                             yerr=random_bidding_sd, fmt='o', label='Random bidding')
            if len(free_rider_bidding_results) > 0:
                plt.errorbar(epochs, free_rider_bidding_average_satisfaction_scores,
                             yerr=free_rider_bidding_sd, fmt='o', label='Free-rider bidding')
            if len(RL_bidding_results) > 0:
                plt.errorbar(epochs, RL_bidder_average_satisfaction_scores,
                             yerr=RL_bidder_sd, fmt='o', label='RL bidding')
        else:
            # Plot the average satisfaction score per epoch, per bidding type (without error bars)
            if len(homogeneous_bidding_results) > 0:
                plt.plot(
                    epochs, homogeneous_bidding_average_satisfaction_scores, 'o', linestyle='None', label='Homogeneous bidding', markersize=1.5)
            if len(heterogeneous_bidding_results) > 0:
                plt.plot(
                    epochs, heterogeneous_bidding_average_satisfaction_scores, 'o', linestyle='None', label='Heterogeneous bidding', markersize=1.5)
            if len(random_bidding_results) > 0:
                plt.plot(
                    epochs, random_bidding_average_satisfaction_scores, 'o', linestyle='None', label='Random bidding', markersize=1.5)
            if len(free_rider_bidding_results) > 0:
                plt.plot(epochs, free_rider_bidding_average_satisfaction_scores,
                         'o', linestyle='None', label='Free-rider bidding', markersize=1.5)
            if len(RL_bidding_results) > 0:
                plt.plot(
                    epochs, RL_bidder_average_satisfaction_scores, 'o', linestyle='None', label='RL bidding', markersize=1.5)
        plt.xlabel('Epoch')
        plt.ylabel('Average Satisfaction Score \n (the higher, the better)')
        plt.title('Average Satisfaction Score per Epoch')
        plt.legend()
        plt.savefig(self.args.results_folder +
                    '/average_satisfaction_score_by_bidding_type.png')
        plt.clf()

    def plot_histogram_satisfaction_scores_by_bidding_type(self, filter_outliers=False):
        """Creates a histogram of all satisfaction scores, over all simulations, for each bidding type, represented by a different color.
        Args:
            export_results (bool): Whether to export the results to a .csv file
            filter_outliers (bool): Whether to filter out outliers from the results
        """
        all_homogeneous_bidding_results = []
        all_heterogeneous_bidding_results = []
        all_random_bidding_results = []
        all_free_rider_bidding_results = []
        all_RL_bidding_results = []

        for result_dict in self.all_simulations_satisfaction_scores:
            for epoch in result_dict:
                for (car_copy, score) in result_dict[epoch]:
                    bidding_type = car_copy.bidding_type
                    # Homogeneous bidding
                    if bidding_type == 'homogeneous':
                        all_homogeneous_bidding_results.append(score)
                    if bidding_type == 'heterogeneous':
                        all_heterogeneous_bidding_results.append(score)
                    # Random bidding
                    elif bidding_type == 'random':
                        all_random_bidding_results.append(score)
                    # Free-rider bidding
                    elif bidding_type == 'free-rider':
                        all_free_rider_bidding_results.append(score)
                    # RL bidding
                    elif bidding_type == 'RL':
                        all_RL_bidding_results.append(score)

        # Remove outliers if necessary:
        if filter_outliers == True:
            if len(all_homogeneous_bidding_results) > 0:
                all_homogeneous_bidding_results = utils.remove_outliers(
                    all_homogeneous_bidding_results)
            if len(all_heterogeneous_bidding_results) > 0:
                all_heterogeneous_bidding_results = utils.remove_outliers(
                    all_heterogeneous_bidding_results)
            if len(all_random_bidding_results) > 0:
                all_random_bidding_results = utils.remove_outliers(
                    all_random_bidding_results)
            if len(all_free_rider_bidding_results) > 0:
                all_free_rider_bidding_results = utils.remove_outliers(
                    all_free_rider_bidding_results)
            if len(all_RL_bidding_results) > 0:
                all_RL_bidding_results = utils.remove_outliers(
                    all_RL_bidding_results)

        # Create a histogram of all satisfaction scores, per bidding type
        if len(all_homogeneous_bidding_results) > 0:
            plt.hist(all_homogeneous_bidding_results, bins=30,
                     alpha=0.5, label='Homogeneous bidding')
        if len(all_heterogeneous_bidding_results) > 0:
            plt.hist(all_heterogeneous_bidding_results, bins=30,
                     alpha=0.5, label='Heterogeneous bidding')
        if len(all_random_bidding_results) > 0:
            plt.hist(all_random_bidding_results, bins=30,
                     alpha=0.5, label='Random bidding')
        if len(all_free_rider_bidding_results) > 0:
            plt.hist(all_free_rider_bidding_results, bins=30,
                     alpha=0.5, label='Free-rider bidding')
        if len(all_RL_bidding_results) > 0:
            plt.hist(all_RL_bidding_results, bins=30,
                     alpha=0.5, label='RL bidding')

        plt.xlabel('Satisfaction Score \n (the lower, the better)')
        plt.ylabel('Frequency \n (all simulations, all epochs)')
        plt.title('Histogram of Satisfaction Scores')
        plt.legend()
        plt.savefig(self.args.results_folder +
                    '/histogram_satisfaction_scores_by_bidding_type.png')
        plt.clf()

    def calc_average_trip_satisfaction(self):
        """Calculates the average trip satisfaction score over all simulations"""

        def remove_car_copies_from_dict(dict):
            """Removes the car copies from the dictionary, so that it only contains the satisfaction scores"""
            return [score for (_, score) in dict]

        all_mean_satisfactions = []
        # First, combine all dictionaries into one dictionary
        for result_dict in self.all_simulations_satisfaction_scores:
            sim_satisfactions = []
            for epoch in result_dict:\
                sim_satisfactions.append(remove_car_copies_from_dict(
                    result_dict[epoch]))
            sim_satisfactions_flat = [
                item for sublist in sim_satisfactions for item in sublist]
            all_mean_satisfactions.append(np.mean(sim_satisfactions_flat))
        
        mean = np.mean(all_mean_satisfactions)
        sd = np.std(all_mean_satisfactions)
        mean_text = str(np.round(mean, 3))
        std_text = str(np.round(sd, 3))
        self.general_metrics['Average Trip Satisfaction'] = str(
            "Mean: " + mean_text + " | SD: " + std_text + " | Description: Average average trip satisfaction. Averaged over sims.")

    ### Congestion Metric ###
    def plot_congestion_heatmap_average(self):
        """Creates a heatmap of the average congestion per epoch per intersection, over all simulations
        Args:
            export_results (bool): Whether to export the results to a .csv file
        """
        # Create heatmap of average congestion per intersection

        total_population_per_intersection = np.sum(
            self.total_population_per_intersection_all_sims, axis=0)
        average_congestion_per_intersection = np.divide(
            total_population_per_intersection, self.args.num_of_simulations - self.num_of_gridlocks)
        average_congestion_per_intersection = np.divide(
            average_congestion_per_intersection, self.args.num_of_epochs)
        average_congestion_per_intersection = np.divide(
            average_congestion_per_intersection, (self.args.queue_capacity * 4))

        ax = sns.heatmap(average_congestion_per_intersection, annot=True)
        ax.set(xlabel='X coordinate', ylabel='Y coordinate',
               title='Average Congestion per Intersection')
        plt.savefig(self.args.results_folder +
                    '/average_congestion_heatmap.png')
        plt.clf()

    def plot_throughput_per_intersection_history(self, export_results=True):
        # Divide by the number of measurements per intersection to calculate the average. If there are no measurements, the average is 0
        total_throughput_history_summed_sims = np.sum(
            self.total_throughput_history_per_intersection_all_sims, axis=0)
        average_throughput_per_intersection = np.divide(
            total_throughput_history_summed_sims, len(self.total_throughput_history_per_intersection_all_sims))
        # Remove the first x epochs from the history, because they are part of the warm-up period
        # Create a plot with subplots for each intersection. Each subplot is a graph of the throughput history of that intersection. In total there are as many subplots as intersections
        fig, axs = plt.subplots(
            average_throughput_per_intersection.shape[0], average_throughput_per_intersection.shape[1], sharex=True, sharey=True, figsize=(20, 20))
        for i in range(average_throughput_per_intersection.shape[0]):
            for j in range(average_throughput_per_intersection.shape[1]):
                axs[i, j].plot(
                    average_throughput_per_intersection[i, j, WARMUP_EPOCHS:], 'o', markersize=1.5)
                axs[i, j].set_title('[' + str(i) + str(j) + ']')
                axs[i, j].set_xlabel('Epoch')
                axs[i, j].set_ylabel('Average Throughput')
        plt.savefig(self.args.results_folder +
                    '/average_throughput_per_intersection_history.png')
        plt.clf()

        if export_results == True:
            np.save(self.export_location + "/average_throughput_per_intersection.npy",
                    average_throughput_per_intersection)

    def calc_central_congestion(self):
        """Calculate the average congestion in the central intersection"""

        average_congestion_per_intersection = np.divide(
            self.total_population_per_intersection_all_sims, self.args.num_of_epochs)
        average_congestion_per_intersection = np.divide(
            average_congestion_per_intersection, (self.args.queue_capacity * 4))

        mean = np.mean(average_congestion_per_intersection, axis=0)
        sd = np.std(average_congestion_per_intersection, axis=0)

        mean_text = str(np.round(mean, 3))
        std_text = str(np.round(sd, 3))
        self.general_metrics['Average congestion per intersection'] = str(
            "Mean: \n" + mean_text + "\nSD:\n" + std_text + "\nDescription: Average congestion per intersection. Averaged over sims")

    ### Winner Worthiness/Auction Reward Metric ###
    def plot_reward_per_intersection_history(self, export_results=True):
        # Divide by the number of measurements per intersection to calculate the average. If there are no measurements, the average is 0
        total_reward_history_summed_sims = np.sum(
            self.reward_history_per_simulation_all_sims, axis=0)
        average_reward_per_intersection = []
        with np.errstate(invalid='ignore'):
            average_reward_per_intersection = np.divide(total_reward_history_summed_sims,
                                                        self.count_of_reward_measurements_per_intersection)
        # Create a plot with subplots for each intersection. Each subplot is a graph of the reward history of that intersection. In total there are as many subplots as intersections
        fig, axs = plt.subplots(
            average_reward_per_intersection.shape[0], average_reward_per_intersection.shape[1], sharex=True, sharey=True, figsize=(20, 20))
        for i in range(average_reward_per_intersection.shape[0]):
            for j in range(average_reward_per_intersection.shape[1]):
                axs[i, j].plot(
                    average_reward_per_intersection[i, j, WARMUP_EPOCHS:], 'o', markersize=1.5)
                axs[i, j].set_title('[' + str(i) + str(j) + ']')
                axs[i, j].set_xlabel('Epoch')
                axs[i, j].set_ylabel('Average Auction Reward')
        plt.savefig(self.args.results_folder +
                    '/average_reward_per_intersection_history.png')
        plt.clf()

        if export_results == True:
            np.save(self.export_location + "/average_reward_per_intersection_history.npy",
                    average_reward_per_intersection)

    def plot_adaptive_auction_parameters_valuations_per_intersection(self, export_results=True):
        """Creates a plot of subplots for each intersection. Each subplot is a 2 or 3d subplot of the evaluation per parameter set."""
        number_of_non_gridlocked_sims = self.args.num_of_simulations - self.num_of_gridlocks
        if NUM_OF_ADAPT_PARAMS == 1:
            # Divide by all the valuations for each parameter set by the number of simulations to calculate the average.
            average_reward_per_parameter_set_per_intersection = np.divide(
                self.sum_auction_parameters_valuations_per_intersection, number_of_non_gridlocked_sims)
            average_count_per_parameter_set_per_intersection = np.divide(
                self.sum_auction_parameters_counts_per_intersection, number_of_non_gridlocked_sims)

            parameter_space_1d = np.reshape(
                self.auction_parameters_space, (self.args.adaptive_auction_discretization, NUM_OF_ADAPT_PARAMS))
            rewards_1d = np.reshape(
                average_reward_per_parameter_set_per_intersection, (self.args.grid_size, self.args.grid_size, self.args.adaptive_auction_discretization))
            counts_1d = np.reshape(
                average_count_per_parameter_set_per_intersection, (self.args.grid_size, self.args.grid_size, self.args.adaptive_auction_discretization))

            # Create a plot of subplots for each intersection. Each subplot is a 2d subplot of the evaluation per parameter set.
            fig, axs = plt.subplots(
                self.args.grid_size, self.args.grid_size, sharex=True, sharey=True, figsize=(20, 20))
            for i in range(self.args.grid_size):
                for j in range(self.args.grid_size):
                    axs[i, j].scatter(
                        parameter_space_1d[:, 0], rewards_1d[i, j, :], s=counts_1d[i, j, :], marker="o")
                    axs[i, j].set_title('[' + str(i) + str(j) + ']')
                    axs[i, j].set_xlabel('Delay Boost')
                    axs[i, j].set_ylabel('Average Reward')
            plt.savefig(self.args.results_folder +
                        '/average_reward_per_parameter_set_per_intersection.png')
            plt.clf()

            if export_results == True:
                np.save(self.export_location + "/average_reward_per_parameter_set_per_intersection_rewards.npy",
                        rewards_1d)
                np.save(self.export_location + "/average_reward_per_parameter_set_per_intersection_parameters.npy",
                        parameter_space_1d)
                np.save(self.export_location + "/average_reward_per_parameter_set_per_intersection_counts.npy",
                        counts_1d)

        elif NUM_OF_ADAPT_PARAMS == 2:
            # Divide by all the valuations for each parameter set by the number of simulations to calculate the average.
            average_reward_per_parameter_set_per_intersection = np.divide(
                self.sum_auction_parameters_valuations_per_intersection, number_of_non_gridlocked_sims)

            parameter_space_2d = np.reshape(
                self.auction_parameters_space, (self.args.adaptive_auction_discretization, self.args.adaptive_auction_discretization, NUM_OF_ADAPT_PARAMS))
            rewards_2d = np.reshape(
                average_reward_per_parameter_set_per_intersection, (self.args.grid_size, self.args.grid_size, self.args.adaptive_auction_discretization, self.args.adaptive_auction_discretization))

            # Create a plot of subplots for each intersection. Each subplot is a 3d subplot of the evaluation per parameter set.
            fig = plt.figure(figsize=(20, 20))
            for i in range(self.args.grid_size):
                for j in range(self.args.grid_size):
                    ax = fig.add_subplot(self.args.grid_size, self.args.grid_size, i *
                                         self.args.grid_size + j + 1, projection='3d')
                    ax.set_title('[' + str(i) + str(j) + ']')
                    ax.set_xlabel('Delay Boost')
                    ax.set_ylabel('QueueLength Boost')
                    ax.set_zlabel('Average Reward')
                    ax.plot_surface(parameter_space_2d[:, :, 0], parameter_space_2d[:, :, 1],
                                    rewards_2d[i, j, :, :], cmap='viridis', edgecolor='none')
            plt.savefig(self.args.results_folder +
                        '/average_reward_per_parameter_set_per_intersection.png')
            plt.clf()

    def plot_mean_bid_and_inact_rank_per_intersection(self):
        mean_bid_rank_per_intersection = np.zeros(
            (self.args.grid_size, self.args.grid_size))
        se_bid_rank_per_intersection = np.zeros(
            (self.args.grid_size, self.args.grid_size))
        for i in range(self.args.grid_size):
            for j in range(self.args.grid_size):
                mean_bid_rank_per_intersection[i, j] = np.mean(
                    self.all_sims_winners_bid_ranks_means[i][j])
                # Below we calculate the SE of the means of the bid ranks per intersection
                se_bid_rank_per_intersection[i, j] = np.std(
                    self.all_sims_winners_bid_ranks_means[i][j]) / np.sqrt(len(self.all_sims_winners_bid_ranks_means[i][j]))

        mean_inact_rank_per_intersection = np.zeros(
            (self.args.grid_size, self.args.grid_size))
        se_inact_rank_per_intersection = np.zeros(
            (self.args.grid_size, self.args.grid_size))
        for i in range(self.args.grid_size):
            for j in range(self.args.grid_size):
                mean_inact_rank_per_intersection[i, j] = np.mean(
                    self.all_sims_winners_inact_ranks_means[i][j])
                se_inact_rank_per_intersection[i, j] = np.std(
                    self.all_sims_winners_inact_ranks_means[i][j]) / np.sqrt(len(self.all_sims_winners_inact_ranks_means[i][j]))

        rank_labels = ['bid_rank', 'inact_rank']
        fig, axs = plt.subplots(
            self.args.grid_size, self.args.grid_size, sharex=True, sharey=True, figsize=(20, 20))
        for i in range(self.args.grid_size):
            for j in range(self.args.grid_size):
                rank_means = [mean_bid_rank_per_intersection[i, j],
                              mean_inact_rank_per_intersection[i, j]]
                rank_ses = [se_bid_rank_per_intersection[i, j],
                            se_inact_rank_per_intersection[i, j]]
                # Create a barplot of the bid ranks per intersection
                axs[i, j].bar(rank_labels, rank_means,
                              yerr=rank_ses, capsize=5)
                axs[i, j].set_title('[' + str(i) + str(j) + ']')
                axs[i, j].set_ylabel('Average Rank')
        plt.savefig(self.args.results_folder +
                    '/winner_bid_inact_rank_per_intersection.png')
        plt.clf()

    ### Time Waited Metric ###
    def calc_time_waited_general_metrics(self):
        # Agent Level
        # Average time waited regardless of intersection. Average over sims
        average_time_waited_per_simulation = []
        for sim in self.average_time_waited_history_per_intersection_all_sims:
            average_time_waited_per_simulation.append(np.mean(sim))
        mean_text = str(round(np.mean(average_time_waited_per_simulation), 3))
        std_text = str(round(np.std(average_time_waited_per_simulation), 3))
        self.general_metrics['agent_average_time_waited'] = str(
            "Mean: " + mean_text + " | SD: " + std_text + " | Description: Average average time waited regardless of intersection. Averaged over sims")
        # Max time waited regardless of intersection. Average over sims
        max_time_waited_per_simulation = []
        for sim in self.max_time_waited_history_per_intersection_all_sims:
            max_time_waited_per_simulation.append(np.mean(sim))
        max_text = str(round(np.mean(max_time_waited_per_simulation), 3))
        std_text = str(round(np.std(max_time_waited_per_simulation), 3))
        self.general_metrics['agent_max_time_waited'] = str(
            "Mean: " + max_text + " | SD: " + std_text + " | Description: Average max time waited regardless of intersection. Averaged over sims")

        # Intersection Level
        # Average time waited, per intersection. Average over sims
        average_time_waited_per_simulation_per_intersection = []
        for sim in self.average_time_waited_history_per_intersection_all_sims:
            average_time_waited_per_simulation_per_intersection.append(
                np.mean(sim, axis=2))
        mean_text = str(np.round(np.mean(
            average_time_waited_per_simulation_per_intersection, axis=0), 3))
        std_text = str(np.round(np.std(
            average_time_waited_per_simulation_per_intersection, axis=0), 3))
        self.general_metrics['intersection_average_time_waited'] = str(
            "Mean: \n" + mean_text + "\nSD:\n" + std_text + "\nDescription: Average average time waited per intersection. Averaged over sims")

        # Max time waited, per intersection. Average over sims
        max_time_waited_per_simulation_per_intersection = []
        for sim in self.max_time_waited_history_per_intersection_all_sims:
            max_time_waited_per_simulation_per_intersection.append(
                np.mean(sim, axis=2))
        max_text = str(np.round(np.mean(
            max_time_waited_per_simulation_per_intersection, axis=0), 3))
        std_text = str(np.round(np.std(
            max_time_waited_per_simulation_per_intersection, axis=0), 3))
        self.general_metrics['intersection_max_time_waited'] = str(
            "Mean: \n" + max_text + "\nSD:\n" + std_text + "\nDescription: Average max time waited per intersection. Averaged over sims")

        # Grid Level
        # Average time waited, aggregated over all intersections. Average over sims.
        average_time_waited_per_simulation_per_intersection = []
        for sim in self.average_time_waited_history_per_intersection_all_sims:
            average_time_waited_per_simulation_per_intersection.append(
                np.mean(sim, axis=2))
        mean_per_intersection = np.mean(
            average_time_waited_per_simulation_per_intersection, axis=0)
        mean_text = str(np.round(np.mean(mean_per_intersection), 3))
        std_text = str(np.round(np.std(mean_per_intersection), 3))
        self.general_metrics['grid_average_time_waited'] = str(
            "Mean: " + mean_text + " | SD: " + std_text + " | Description: Average average time waited of averages of intersections. Averaged over intersections")

        # Max time waited, aggregated over all intersections. Average over sims.
        max_time_waited_per_simulation_per_intersection = []
        for sim in self.max_time_waited_history_per_intersection_all_sims:
            max_time_waited_per_simulation_per_intersection.append(
                np.mean(sim, axis=2))
        max_per_intersection = np.mean(
            max_time_waited_per_simulation_per_intersection, axis=0)
        max_text = str(np.round(np.mean(max_per_intersection), 3))
        std_text = str(np.round(np.std(max_per_intersection), 3))
        self.general_metrics['grid_max_time_waited'] = str(
            "Mean: " + max_text + " | SD: " + std_text + " | Description: Average max time waited of averages of intersections. Averaged over intersections")

    def calc_time_waited_gini_metric(self):
        average_gini_time_waited_history_sims = np.mean(
            self.gini_time_waited_history_per_intersection_all_sims, axis=3)  # Average over epochs
        mean_gini_per_simulation = []
        for sim in average_gini_time_waited_history_sims:
            mean_gini_per_simulation.append(
                np.mean(sim))  # Average over intersections
        mean_gini_text = str(round(np.mean(mean_gini_per_simulation), 3))
        std_gini_text = str(round(np.std(mean_gini_per_simulation), 3))
        self.general_metrics['time_waited_avg_gini'] = str(
            "Mean: " + mean_gini_text + " | SD: " + std_gini_text + " | Description: The average of the GINI coefficients of all intersections. Averaged over sims")

    def plot_average_time_waited_per_intersection_history(self, export_results=True):
        # The first x epochs are part of the warm-up period, so they are not included in the results
        # Divide by the number of measurements per intersection to calculate the average. If there are no measurements, the average is 0
        total_average_time_waited_history_summed_sims = np.sum(
            self.average_time_waited_history_per_intersection_all_sims, axis=0)
        average_average_time_waited_per_intersection = []
        with np.errstate(invalid='ignore'):
            average_average_time_waited_per_intersection = np.divide(
                total_average_time_waited_history_summed_sims, self.count_of_average_time_waited_measurements_per_intersection)
        # Create a plot with subplots for each intersection. Each subplot is a graph of the average_time_waited history of that intersection. In total there are as many subplots as intersections
        fig, axs = plt.subplots(
            average_average_time_waited_per_intersection.shape[0], average_average_time_waited_per_intersection.shape[1], sharex=True, sharey=True, figsize=(20, 20))
        for i in range(average_average_time_waited_per_intersection.shape[0]):
            for j in range(average_average_time_waited_per_intersection.shape[1]):
                axs[i, j].plot(
                    average_average_time_waited_per_intersection[i, j, WARMUP_EPOCHS:], 'o', markersize=1.5)
                axs[i, j].set_title('[' + str(i) + str(j) + ']')
                axs[i, j].set_xlabel('Epoch')
                axs[i, j].set_ylabel('Average Average Time Waited')
        plt.savefig(self.args.results_folder +
                    '/average_average_time_waited_per_intersection_history.png')
        plt.clf()
        if export_results == True:
            np.save(self.export_location + "/average_average_time_waited_per_intersection_history.npy",
                    average_average_time_waited_per_intersection)

    def plot_max_time_waited_per_intersection_history(self, export_results=True):
        # The first x epochs are part of the warm-up period, so they are not included in the results
        # Divide by the number of measurements per intersection to calculate the average. If there are no measurements, the average is 0
        total_max_time_waited_history_summed_sims = np.sum(
            self.max_time_waited_history_per_intersection_all_sims, axis=0)
        average_max_time_waited_per_intersection = []
        with np.errstate(invalid='ignore'):
            average_max_time_waited_per_intersection = np.divide(
                total_max_time_waited_history_summed_sims, self.count_of_max_time_waited_measurements_per_intersection)
        # Create a plot with subplots for each intersection. Each subplot is a graph of the max_time_waited history of that intersection. In total there are as many subplots as intersections
        fig, axs = plt.subplots(
            average_max_time_waited_per_intersection.shape[0], average_max_time_waited_per_intersection.shape[1], sharex=True, sharey=True, figsize=(20, 20))
        for i in range(average_max_time_waited_per_intersection.shape[0]):
            for j in range(average_max_time_waited_per_intersection.shape[1]):
                axs[i, j].plot(
                    average_max_time_waited_per_intersection[i, j, WARMUP_EPOCHS:], 'o', markersize=1.5)
                axs[i, j].set_title('[' + str(i) + str(j) + ']')
                axs[i, j].set_xlabel('Epoch')
                axs[i, j].set_ylabel('Average Max Time Waited')
        plt.savefig(self.args.results_folder +
                    '/average_max_time_waited_per_intersection_history.png')
        plt.clf()
        if export_results == True:
            np.save(self.export_location + "/average_max_time_waited_per_intersection_history.npy",
                    average_max_time_waited_per_intersection)

    def plot_gini_time_waited_per_intersection_history(self, export_results=True):
        # The first x epochs are part of the warm-up period, so they are not included in the results
        # Divide by the number of measurements per intersection to calculate the average. If there are no measurements, the average is 0
        total_gini_time_waited_history_summed_sims = np.sum(
            self.gini_time_waited_history_per_intersection_all_sims, axis=0)
        average_gini_time_waited_per_intersection = []
        with np.errstate(invalid='ignore'):
            average_gini_time_waited_per_intersection = np.divide(
                total_gini_time_waited_history_summed_sims, self.count_of_gini_time_waited_measurements_per_intersection)
        # Create a plot with subplots for each intersection. Each subplot is a graph of the gini history of that intersection. In total there are as many subplots as intersections
        fig, axs = plt.subplots(
            average_gini_time_waited_per_intersection.shape[0], average_gini_time_waited_per_intersection.shape[1], sharex=True, sharey=True, figsize=(20, 20))
        for i in range(average_gini_time_waited_per_intersection.shape[0]):
            for j in range(average_gini_time_waited_per_intersection.shape[1]):
                axs[i, j].plot(
                    average_gini_time_waited_per_intersection[i, j, WARMUP_EPOCHS:], 'o', markersize=1.5)
                axs[i, j].set_title('[' + str(i) + str(j) + ']')
                axs[i, j].set_xlabel('Epoch')
                axs[i, j].set_ylabel(
                    'Average GINI Coefficient (based on time waited)')
        plt.savefig(self.args.results_folder +
                    '/average_gini_time_waited_per_intersection_history.png')
        plt.clf()
        if export_results == True:
            np.save(self.export_location + "/average_gini_time_waited_per_intersection_history.npy",
                    average_gini_time_waited_per_intersection)

    ### Misc. Metrics ###
    def plot_broke_agents_percentage_history(self):
        """Plot a history of the average percentage of agents that have a balance of 0"""
        avg_percentage_broke_agents = np.divide(
            self.all_sims_broke_agents_history, self.args.num_of_simulations)

        # Create a single plot for the entire grid
        # make the plot font quite big
        plt.rcParams.update({'font.size': 30})
        plt.plot(avg_percentage_broke_agents)
        plt.title('Average Percentage of Agents with 0 balance over time \n (adaptive bidders)')
        plt.xlabel('Epoch')
        plt.ylim(0, 0.5)
        plt.ylabel('Average Percentage of Agents with 0 balance')
        plt.savefig(self.args.results_folder +
                    '/average_percentage_broke_agents_history.png')
        plt.clf()

    def calc_average_num_of_trips_completed(self):
        num_of_trips_per_simulation = []
        for sim in self.all_simulations_satisfaction_scores:
            num_of_trips_in_sim = 0  # That disregards the epoch
            for epoch in sim:
                if sim[epoch] != None:
                    num_of_trips_in_sim += len(sim[epoch])
            num_of_trips_per_simulation.append(num_of_trips_in_sim)

        mean_text = str(round(np.mean(num_of_trips_per_simulation), 3))
        std_text = str(round(np.std(num_of_trips_per_simulation), 3))

        self.general_metrics['num_of_trips_completed'] = str(
            "Mean: " + mean_text + " | SD: " + std_text + " | Description: The mean number of trips completed per simulation. Averaged over sims")

class SimulationMetrics:
    """
    The SimulationMetrics class is responsible for keeping track of all the metrics for a single simulation.
    Attributes:
        args (argparse.Namespace): Arguments parsed from the command line
        grid (Grid): The grid object of the simulation
        current_sim_satisfaction_scores (dict): A dictionary of satisfaction scores for each car. The key is the epoch, the value is
            a list of all the satisfaction scores for that epoch, if any.
        total_throughput_per_intersection (dict): A dictionary of the total throughput of each intersection. The key is the intersection id,
            the value is the total throughput.
        throughput_history_per_intersection (np.array): A 3d array of the throughput history of each intersection. The first index is the x coordinate,
            the second is the y coordinate, the third is the epoch. The value is the throughput.
        reward_history_per_intersection (np.array): A 3d array of the reward history of each intersection. The first index is the x coordinate,
            the second is the y coordinate, the third is the epoch. The value is the reward.
        max_time_waited_history_per_intersection (np.array): A 3d array of the max time waited history of each intersection. The first index is the x coordinate,
            the second is the y coordinate, the third is the epoch. The value is the max time waited for that intersection at that epoch.
        auction_parameters_space (np.array): A 2d array of the auction parameters space. The first index is the parameter set, the second is the parameter.
        auction_parameters_valuations_per_intersection (list): A 2d list of the valuations of each parameter set for each intersection. The first index is the x coordinate,
            the second is the y coordinate. The value is a list of the valuations of each parameter set for that intersection.
        auction_parameters_counts_per_intersection (list): A 2d list of the counts of each parameter set for each intersection. The first index is the x coordinate,
            the second is the y coordinate. The value is a list of the counts (number of trials) of each parameter set for that intersection.
    Functions:
        add_satisfaction_scores(epoch, satisfaction_scores): Adds the satisfaction scores of the cars that completed a trip.
            If there was no car that completed a trip in an epoch, there is no entry for that epoch.
        ready_for_new_epoch(): Prepares the metrics keeper for the next epoch
        retrieve_end_of_simulation_metrics(): Retrieves the metrics at the end of the simulation
        check_if_gridlocked(): Checks if the simulation is gridlocked
    """

    def __init__(self, args, grid):
        """ Initialize the MetricsKeeper object
        Args:
            args (argparse.Namespace): Arguments parsed from the command line
            grid (Grid): The grid object of the simulation
        """
        self.args = args
        self.grid = grid

        ### Satisfaction Metric ###
        self.current_sim_satisfaction_scores = {}

        ### Congestion Metric ###
        self.total_population_per_intersection = np.zeros(
            (args.grid_size, args.grid_size))

        self.throughput_history_per_intersection = np.zeros(
            (args.grid_size, args.grid_size, args.num_of_epochs))

        ### Time Waited Metrics ###
        self.average_time_waited_history_per_intersection = np.zeros(
            (args.grid_size, args.grid_size, args.num_of_epochs))

        self.max_time_waited_history_per_intersection = np.zeros(
            (args.grid_size, args.grid_size, args.num_of_epochs))

        self.gini_time_waited_history_per_intersection = np.zeros(
            (args.grid_size, args.grid_size, args.num_of_epochs))

        ### Auction Metrics ###
        self.reward_history_per_intersection = np.zeros(
            (args.grid_size, args.grid_size, args.num_of_epochs))
        # The bid and inact ranks have 2 lists each, one for mean and one for the standard errors
        self.winners_inact_ranks_per_intersection_means = np.zeros(
            (args.grid_size, args.grid_size))
        self.winners_inact_ranks_per_intersection_ses = np.zeros(
            (args.grid_size, args.grid_size))
        self.winners_bid_ranks_per_intersection_means = np.zeros(
            (args.grid_size, args.grid_size))
        self.winners_bid_ranks_per_intersection_ses = np.zeros(
            (args.grid_size, args.grid_size))
        # If discretisation is adaptive_auction_discretization and there are 2 parameters,
        # then there are adaptive_auction_discretization^2 possible combinations of parameters
        self.auction_parameters_space = np.zeros(
            (pow(args.adaptive_auction_discretization, NUM_OF_ADAPT_PARAMS), NUM_OF_ADAPT_PARAMS))
        self.auction_parameters_valuations_per_intersection = [
            [[] for _ in range(args.grid_size)] for _ in range(args.grid_size)]
        self.auction_parameters_counts_per_intersection = [
            [[] for _ in range(args.grid_size)] for _ in range(args.grid_size)]

        ### Misc. ###
        self.broke_history = []

    def check_if_gridlocked(self):
        # If a queue has been inactive for more than half of the simulation, it is considered gridlocked
        if (np.nanmax(list(np.concatenate(self.max_time_waited_history_per_intersection).flat)) > self.args.num_of_epochs / 2):
            return True

        return False

    def add_satisfaction_scores(self, epoch, satisfaction_scores):
        """Adds the satisfaction scores of the cars that completed a trip. If there was no car that completed
            a trip in an epoch, there is no entry for that epoch.
        Args:
            epoch (int): The epoch in which the cars completed their trip
            satisfaction_scores (list): A list of tuples, containing small car copies and their satisfaction scores of the completed trip
        """
        if satisfaction_scores:  # If it is not empty
            self.current_sim_satisfaction_scores[epoch] = satisfaction_scores

    def ready_for_new_epoch(self):
        """Prepares the metrics keeper for the next epoch"""
        # We use a 2d array. The first index is the x coordinate, the second is the y coordinate.
        # Here, we store all the total throughput per intersection and the last reward per intersection
        for intersection in self.grid.all_intersections:
            id = intersection.id
            x_cord, y_cord = map(int, id)
            self.total_population_per_intersection[x_cord][y_cord] += intersection.get_num_of_cars_in_intersection()

    def retrieve_end_of_simulation_metrics(self):
        """Retrieves the metrics at the end of the simulation"""
        for intersection in self.grid.all_intersections:
            id = intersection.id
            x_cord, y_cord = map(int, id)

            ### Time Waited Metrics ###
            # Gather the average_time_waited history of each intersection
            self.average_time_waited_history_per_intersection[x_cord][y_cord] = intersection.get_average_time_waited_history(
            )
            # Gather the max_time_waited history of each intersection
            self.max_time_waited_history_per_intersection[x_cord][y_cord] = intersection.get_max_time_waited_history(
            )
            # Gather the gini time waited coeffs of each intersection
            self.gini_time_waited_history_per_intersection[x_cord][y_cord] = intersection.get_gini_time_waited_history(
            )

            ### Congestion Metric ###
            # Gather the throughput history of each intersection
            self.throughput_history_per_intersection[x_cord][y_cord] = intersection.get_auction_throughput_history(
            )

            ### Auction Metrics ###
            # Gather the reward history of each intersection
            self.reward_history_per_intersection[x_cord][y_cord] = intersection.get_auction_reward_history(
            )
            # Gather the auction parameters and their valuations of each intersection. The parameter space is the same for all intersections
            self.auction_parameters_space, self.auction_parameters_valuations_per_intersection[x_cord][y_cord], self.auction_parameters_counts_per_intersection[x_cord][y_cord] = intersection.get_auction_parameters_and_valuations_and_counts(
            )
            # Gather winner mean & se bid and inact ranks for each itnersection:
            self.winners_inact_ranks_per_intersection_means[x_cord][y_cord] = intersection.calc_and_get_mean_winners_inact_ranks(
            )
            self.winners_bid_ranks_per_intersection_means[x_cord][y_cord] = intersection.calc_and_get_mean_winners_bid_ranks(
            )

        ### Misc. ###
        self.broke_history = self.grid.get_broke_history()