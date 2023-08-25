"""A class to keep track of the metrics of the simulation and create relevant graphs."""
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from math import nan
import pandas as pd
import csv
from itertools import zip_longest

import src.utils as utils

NUM_OF_ADAPT_PARAMS = 1
WARMUP_EPOCHS = 30


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

    def __init__(self, args):
        """ Initialize the MetricsKeeper object
        Args:
            args (argparse.Namespace): Arguments parsed from the command line
        """
        self.args = args

        self.export_location = self.args.results_folder + "/exported_data"
        if not os.path.exists(self.export_location):
            os.makedirs(self.export_location)

        # Number of Gridlocked simulations
        self.num_of_gridlocks = 0

        # The satisfaction scores history of all simulations
        self.all_simulations_satisfaction_scores = []

        # Total throughput per intersection
        self.total_population_per_intersection_all_sims = []

        # Total throughput history per intersection
        self.total_throughput_history_per_intersection_all_sims = []

        # Number of reward measurements per intersection, used to calculate the average
        self.count_of_reward_measurements_per_intersection = np.zeros(
            (self.args.grid_size, self.args.grid_size, self.args.num_of_epochs))

        # Total reward history per intersection
        self.reward_history_per_simulation_all_sims = []

        # Number of reward measurements per intersection, used to calculate the average
        self.count_of_max_time_waited_measurements_per_intersection = np.zeros(
            (self.args.grid_size, self.args.grid_size, self.args.num_of_epochs))

        # Max time waited history per intersection
        self.max_time_waited_history_per_intersection_all_sims = []

        # Auction parameter space
        self.auction_parameters_space = []

        # Sum of auction parameters valuations per intersection
        self.sum_auction_parameters_valuations_per_intersection = np.zeros(
            (self.args.grid_size, self.args.grid_size, pow(self.args.adaptive_auction_discretization, NUM_OF_ADAPT_PARAMS)))

        # Count of auction parameters trials per intersection
        self.sum_auction_parameters_counts_per_intersection = np.zeros(
            (self.args.grid_size, self.args.grid_size, pow(self.args.adaptive_auction_discretization, NUM_OF_ADAPT_PARAMS)))

    def store_simulation_results(self, sim_metrics_keeper):
        """Prepares the metrics keeper for a new simulation, by clearing the results of the current simulation
        Args:
            sim_metrics_keeper (SimulationMetrics): The metrics keeper of the current simulation
        """

        if sim_metrics_keeper.check_if_gridlocked() == True:
            # If the simulation was gridlocked, take note of that and do not store the results.
            self.num_of_gridlocks += 1
            return

        self.all_simulations_satisfaction_scores.append(
            sim_metrics_keeper.current_sim_satisfaction_scores)

        self.total_population_per_intersection_all_sims.append(
            sim_metrics_keeper.total_population_per_intersection)

        self.total_throughput_history_per_intersection_all_sims.append(
            sim_metrics_keeper.throughput_history_per_intersection)

        # For each measurement that is not nan, we add 1 to the count of measurements, so that we can later calculate the average
        self.count_of_reward_measurements_per_intersection += np.where(
            np.isnan(sim_metrics_keeper.reward_history_per_intersection), 0, 1)
        self.reward_history_per_simulation_all_sims.append(
            np.nan_to_num(sim_metrics_keeper.reward_history_per_intersection))

        self.count_of_max_time_waited_measurements_per_intersection += np.where(
            np.isnan(sim_metrics_keeper.max_time_waited_history_per_intersection), 0, 1)
        self.max_time_waited_history_per_intersection_all_sims.append(np.nan_to_num(
            sim_metrics_keeper.max_time_waited_history_per_intersection))

        # Retrieve the parameter space and the valuations per intersection. The parameters space is the same for all intersections
        self.auction_parameters_space = sim_metrics_keeper.auction_parameters_space

        self.sum_auction_parameters_valuations_per_intersection += sim_metrics_keeper.auction_parameters_valuations_per_intersection
        self.sum_auction_parameters_counts_per_intersection += sim_metrics_keeper.auction_parameters_counts_per_intersection

    def produce_results(self):
        """Produces all the evaluation results of all simulations"""

        # Create a .txt file with the arguments used for the simulation
        with open(self.args.results_folder + '/configuration.txt', 'w') as f:
            for arg in vars(self.args):
                f.write(arg + ': ' + str(getattr(self.args, arg)) + '\n')

        # Produce all the general metrics
        self.produce_general_metrics()

        # Create a graph of all satisfaction scores, over all simulations
        self.plot_satisfaction_scores_overall_average()

        # Create a graph of all satisfaction scores, per bidding type, over all simulations
        self.plot_satisfaction_scores_by_bidding_type()

        # Create a histogram of all satisfaction scores, over all simulations, per bidding type
        self.histogram_satisfaction_scores_by_bidding_type()

        # Create a heatmap of the average throughput per intersection, over all simulations
        self.plot_congestion_heatmap_average()

        # Create a graph with graphs of the average throughput per intersection, over all simulations
        self.plot_throughput_per_intersection_history()

        # Create a graph with graphs of the average max time waited per intersection, over all simulations
        self.plot_max_time_waited_per_intersection_history()

        # Create a graph with graphs of the average reward per intersection, over all simulations
        self.plot_reward_per_intersection_history()

        self.plot_adaptive_auction_parameters_valuations_per_intersection()

    def produce_general_metrics(self):
        """Produces the general metrics of all simulations"""
        # 1st: Calculate the last-epoch average throughput per intersection over all simulations.
        # Only keep the last epoch
        throughput_only_last_epoch_all_sims = []
        for sim in self.total_throughput_history_per_intersection_all_sims:
            throughput_only_last_epoch_all_sims.append(sim[:, :, -1])
        throughput_only_last_epoch_all_sims = np.array(
            throughput_only_last_epoch_all_sims)

        throughput_only_last_epoch_all_sims_sum = np.sum(
            throughput_only_last_epoch_all_sims, axis=0)
        # throughput_only_last_epoch_all_sims.shape[0] is the number of simulations, excluding grilocks.
        throughput_only_last_epoch_all_sims_avg = np.divide(
            throughput_only_last_epoch_all_sims_sum, throughput_only_last_epoch_all_sims.shape[0])
        # Calculate the standard deviation of throughput_only_last_epoch_all_sims over all sims.
        last_epoch_average_throughput_std = np.std(
            throughput_only_last_epoch_all_sims, axis=0)
        # Dictionary for each intersection, holding the average throughput and the SD for printing
        throughput_per_intersection_last_epoch_dict = {}
        for i in range(self.args.grid_size):
            for j in range(self.args.grid_size):
                throughput_per_intersection_last_epoch_dict[str(i) + str(j)] = str(
                    str(throughput_only_last_epoch_all_sims_avg[i, j]) + " (SD: " + str(round(last_epoch_average_throughput_std[i, j], 2)) + ")")

        # 2nd: Calculate the last-epoch average reward over all intersections and all simulations.
        total_reward_only_last_epoch_all_sims = []
        for sim in self.reward_history_per_simulation_all_sims:
            total_reward_only_last_epoch_all_sims.append(sim[:, :, -1])
        total_reward_only_last_epoch_all_sims = np.array(
            total_reward_only_last_epoch_all_sims)

        total_reward_only_last_epoch_all_sims_sum = np.sum(
            total_reward_only_last_epoch_all_sims, axis=0)
        total_reward_only_last_epoch_all_sims_avg = np.divide(
            total_reward_only_last_epoch_all_sims_sum, total_reward_only_last_epoch_all_sims.shape[0])
        # Calculate the standard deviation
        last_epoch_average_total_reward_std = np.std(
            total_reward_only_last_epoch_all_sims, axis=0)
        # Dictionary for each intersection, holding the average reward and the SD for printing
        total_reward_per_intersection_last_epoch_dict = {}
        for i in range(self.args.grid_size):
            for j in range(self.args.grid_size):
                total_reward_per_intersection_last_epoch_dict[str(i) + str(j)] = str(
                    str(round(total_reward_only_last_epoch_all_sims_avg[i, j], 2)) + " (SD: " + str(round(last_epoch_average_total_reward_std[i, j], 2)) + ")")

        # 3rd: Calculate the average total reward of each intersection
        average_epoch_reward_per_intersection_all_sims = []
        for sim in self.reward_history_per_simulation_all_sims:
            average_epoch_reward_per_intersection_all_sims.append(
                np.sum(sim, axis=2))
        average_epoch_reward_per_intersection_all_sims = np.array(
            average_epoch_reward_per_intersection_all_sims)
        average_epoch_reward_per_intersection_all_sims = np.divide(
            average_epoch_reward_per_intersection_all_sims, self.args.num_of_epochs)
        average_epoch_reward_per_intersection_all_sims_std = np.std(
            average_epoch_reward_per_intersection_all_sims, axis=0)
        average_epoch_reward_per_intersection_all_sims = np.sum(
            average_epoch_reward_per_intersection_all_sims, axis=0)
        average_epoch_reward_per_intersection_all_sims = np.divide(
            average_epoch_reward_per_intersection_all_sims, len(self.reward_history_per_simulation_all_sims))
        average_epoch_reward_per_intersection_dic = {}
        for i in range(self.args.grid_size):
            for j in range(self.args.grid_size):
                average_epoch_reward_per_intersection_dic[str(i) + str(j)] = str(
                    str(round(average_epoch_reward_per_intersection_all_sims[i, j], 4)) + " (SD: " + str(round(average_epoch_reward_per_intersection_all_sims_std[i, j], 4)) + ")")

        # 3rd: Calculate the last-epoch average max time waited over all intersections and all simulations.
        total_max_time_waited_only_last_epoch_all_sims = []
        for sim in self.max_time_waited_history_per_intersection_all_sims:
            total_max_time_waited_only_last_epoch_all_sims.append(
                sim[:, :, -1])
        total_max_time_waited_only_last_epoch_all_sims = np.array(
            total_max_time_waited_only_last_epoch_all_sims)

        total_max_time_waited_only_last_epoch_all_sims_sum = np.sum(
            total_max_time_waited_only_last_epoch_all_sims, axis=0)
        total_max_time_waited_only_last_epoch_all_sims_avg = np.divide(
            total_max_time_waited_only_last_epoch_all_sims_sum, len(total_max_time_waited_only_last_epoch_all_sims))
        # Calculate the standard deviation
        total_max_time_waited_only_last_epoch_all_sims_std = np.std(
            total_max_time_waited_only_last_epoch_all_sims, axis=0)
        # Dictionary for each intersection, holding the average max time waited and the SD for printing
        total_max_time_waited_per_intersection_last_epoch_dict = {}
        for i in range(self.args.grid_size):
            for j in range(self.args.grid_size):
                total_max_time_waited_per_intersection_last_epoch_dict[str(i) + str(j)] = str(
                    str(round(total_max_time_waited_only_last_epoch_all_sims_avg[i, j], 2)) + " (SD: " + str(round(total_max_time_waited_only_last_epoch_all_sims_std[i, j], 2)) + ")")

        # 4th: Calculate the last-epoch average satisfaction score over all simulations.
        last_epoch_satisfaction_scores_all_sims = []
        for sim_dict in self.all_simulations_satisfaction_scores:
            # This gets the last epoch measurements of each simulation
            for (car_copy, score) in sim_dict[list(sim_dict)[-1]]:
                last_epoch_satisfaction_scores_all_sims.append(score)
        last_epoch_satisfaction_scores_avg = sum(
            last_epoch_satisfaction_scores_all_sims) / len(last_epoch_satisfaction_scores_all_sims)
        last_epoch_satisfaction_scores_std = np.std(
            last_epoch_satisfaction_scores_all_sims)

        # 5th: Calculate the last-epoch average satisfaction score per bidding type, over all simulations.
        last_epoch_average_satisfaction_score_per_bidding_type = {}
        last_epoch_average_satisfaction_score_per_bidding_type_std = {}
        # Holds all the cars that completed a trip on the last epoch.
        all_last_sim_cars = []
        for sim_dict in self.all_simulations_satisfaction_scores:
            # This gets the last epoch measurements of each simulation
            for (car_copy, score) in sim_dict[list(sim_dict)[-1]]:
                all_last_sim_cars.append((car_copy, score))
        scores_per_bidding_type = {}
        for (car_copy, score) in all_last_sim_cars:
            bidding_type = car_copy.bidding_type
            if bidding_type in scores_per_bidding_type:
                scores_per_bidding_type[bidding_type].append(score)
            else:
                scores_per_bidding_type[bidding_type] = [score]
        # The average is now calculated
        for bidding_type in scores_per_bidding_type:
            last_epoch_average_satisfaction_score_per_bidding_type[bidding_type] = round(sum(
                scores_per_bidding_type[bidding_type]) / len(scores_per_bidding_type[bidding_type]), 2)
        # Calculate the standard deviation
        last_epoch_average_satisfaction_score_per_bidding_type_std = {}
        for bidding_type in scores_per_bidding_type:
            last_epoch_average_satisfaction_score_per_bidding_type_std[bidding_type] = round(np.std(
                scores_per_bidding_type[bidding_type]), 2)

        # Export the results to a .txt file
        with open(self.args.results_folder + '/general_metrics.txt', 'w') as f:
            f.write('Num of excluded simulations, due to gridlocks: ' +
                    str(self.num_of_gridlocks) + '\n')
            f.write('Last-epoch average throughput over all intersections and all simulations: ' + str(
                throughput_per_intersection_last_epoch_dict) + '\n')
            f.write('Last-epoch average throughput over all intersections and all simulations: ' + str(
                total_reward_per_intersection_last_epoch_dict) + '\n')
            f.write('Last-epoch average max_time_waited over all intersections and all simulations: ' + str(
                total_max_time_waited_per_intersection_last_epoch_dict) + '\n')
            f.write('Last-epoch average satisfaction score over all simulations: ' +
                    str(round(last_epoch_satisfaction_scores_avg, 2)) + " (SD: " +
                    str(round(last_epoch_satisfaction_scores_std, 2)) + ')\n')
            f.write('Last-epoch average satisfaction score per bidding type, over all simulations: ' +
                    str(last_epoch_average_satisfaction_score_per_bidding_type) + " (SD: " + str(last_epoch_average_satisfaction_score_per_bidding_type_std) + ')\n')
            f.write('Average epoch reward per intersection:' + str(
                average_epoch_reward_per_intersection_dic) + '\n')

    def plot_satisfaction_scores_overall_average(self, export_results=True):
        """Creates a graph of the average satisfaction score per epoch, with error bars, averaged over all simulations."""

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

        # Plot the average satisfaction score per epoch, without error bars
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

    def plot_satisfaction_scores_by_bidding_type(self, with_std=False, export_results=True, filter_outliers=False):
        """Creates a graph of the average satisfaction score per epoch, with error bars, averaged over all simulations,
            for each bidding type, represented by a different color.
            'ohhh almost 200 lines of code, that's a lot of code for just one function (but here we are)'
        Args:
            with_std (bool): Whether to include the standard deviation in the plot
            export_results (bool): Whether to export the results to a .csv file
            filter_outliers (bool): Whether to filter out outliers from the results
        """

        static_low_bidding_results = {}
        static_high_bidding_results = {}
        random_bidding_results = {}
        free_rider_bidding_results = {}
        RL_bidding_results = {}

        for result_dict in self.all_simulations_satisfaction_scores:
            for epoch in result_dict:
                for (car_copy, score) in result_dict[epoch]:
                    bidding_type = car_copy.bidding_type
                    # Static bidding
                    if bidding_type == 'static_low':
                        if epoch in static_low_bidding_results:
                            static_low_bidding_results[epoch].append(score)
                        else:
                            static_low_bidding_results[epoch] = [score]
                    elif bidding_type == 'static_high':
                        if epoch in static_high_bidding_results:
                            static_high_bidding_results[epoch].append(score)
                        else:
                            static_high_bidding_results[epoch] = [score]
                    # Random bidding
                    elif bidding_type == 'random':
                        if epoch in random_bidding_results:
                            random_bidding_results[epoch].append(score)
                        else:
                            random_bidding_results[epoch] = [score]
                    # Free-rider bidding
                    elif bidding_type == 'free-rider':
                        if epoch in free_rider_bidding_results:
                            free_rider_bidding_results[epoch].append(score)
                        else:
                            free_rider_bidding_results[epoch] = [score]
                    # RL bidding
                    elif bidding_type == 'RL':
                        if epoch in RL_bidding_results:
                            RL_bidding_results[epoch].append(score)
                        else:
                            RL_bidding_results[epoch] = [score]

        # Create a list of all epochs in which cars completed their trip
        epochs = []
        for epoch in static_low_bidding_results:
            if static_low_bidding_results[epoch] != None or static_high_bidding_results[epoch] != None or random_bidding_results[epoch] != None or free_rider_bidding_results[epoch] != None or RL_bidding_results[epoch] != None:
                epochs.append(epoch)

        # Remove outliers if necessary:
        if filter_outliers == True:
            for epoch in epochs:
                # Static low bidding:
                if epoch in static_low_bidding_results:
                    static_low_bidding_results[epoch] = utils.remove_outliers(
                        static_low_bidding_results[epoch])
                # Static high bidding
                if epoch in static_high_bidding_results:
                    static_high_bidding_results[epoch] = utils.remove_outliers(
                        static_high_bidding_results[epoch])
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
        static_low_bidding_average_satisfaction_scores = []
        static_high_bidding_average_satisfaction_scores = []
        random_bidding_average_satisfaction_scores = []
        free_rider_bidding_average_satisfaction_scores = []
        RL_bidder_average_satisfaction_scores = []

        for epoch in epochs:
            # Static low bidding:
            if epoch in static_low_bidding_results:
                static_low_bidding_average_satisfaction_scores.append(
                    sum(static_low_bidding_results[epoch]) / len(static_low_bidding_results[epoch]))
            else:
                static_low_bidding_average_satisfaction_scores.append(nan)
            # Static high bidding:
            if epoch in static_high_bidding_results:
                static_high_bidding_average_satisfaction_scores.append(
                    sum(static_high_bidding_results[epoch]) / len(static_high_bidding_results[epoch]))
            else:
                static_high_bidding_average_satisfaction_scores.append(nan)
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
        static_low_bidding_sd = []
        static_high_bidding_sd = []
        random_bidding_sd = []
        free_rider_bidding_sd = []
        RL_bidder_sd = []
        for epoch in epochs:
            # Static low bidding:
            if epoch in static_low_bidding_results:
                static_low_bidding_sd.append(
                    np.std(static_low_bidding_results[epoch]))
            else:
                static_low_bidding_sd.append(nan)
            # Static high bidding:
            if epoch in static_high_bidding_results:
                static_high_bidding_sd.append(
                    np.std(static_high_bidding_results[epoch]))
            else:
                static_high_bidding_sd.append(nan)
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
        static_low_bidding_average_satisfaction_scores = static_low_bidding_average_satisfaction_scores[
            WARMUP_EPOCHS:]
        static_low_bidding_sd = static_low_bidding_sd[WARMUP_EPOCHS:]

        static_high_bidding_average_satisfaction_scores = static_high_bidding_average_satisfaction_scores[
            WARMUP_EPOCHS:]
        static_high_bidding_sd = static_high_bidding_sd[WARMUP_EPOCHS:]

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
            if len(static_low_bidding_results) > 0:
                plt.errorbar(epochs, static_low_bidding_average_satisfaction_scores,
                             yerr=static_low_bidding_sd, fmt='o', label='Static low bidding')
            if len(static_high_bidding_results) > 0:
                plt.errorbar(epochs, static_high_bidding_average_satisfaction_scores,
                             yerr=static_high_bidding_sd, fmt='o', label='Static high bidding')
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
            if len(static_low_bidding_results) > 0:
                plt.plot(
                    epochs, static_low_bidding_average_satisfaction_scores, 'o', linestyle='None', label='Static low bidding', markersize=1.5)
            if len(static_high_bidding_results) > 0:
                plt.plot(
                    epochs, static_high_bidding_average_satisfaction_scores, 'o', linestyle='None', label='Static high bidding', markersize=1.5)
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

        if export_results == True:
            np.savetxt(self.args.results_folder + '/average_satisfaction_score_by_bidding_type.csv', np.array([
                epochs, static_low_bidding_average_satisfaction_scores, static_low_bidding_sd,
                static_high_bidding_average_satisfaction_scores, static_high_bidding_sd,
                random_bidding_average_satisfaction_scores, random_bidding_sd,
                free_rider_bidding_average_satisfaction_scores, free_rider_bidding_sd,
                RL_bidder_average_satisfaction_scores, RL_bidder_sd
            ]).T, delimiter=",", header="Epoch, Static low bidding Score, Static low bidding SD, Static high bidding Score, Static high bidding SD, Random bidding Score, Random bidding SD, Free-rider bidding Score, Free-rider bidding SD, RL bidding Score, RL bidding SD")

    def histogram_satisfaction_scores_by_bidding_type(self, export_results=True, filter_outliers=False):
        """Creates a histogram of all satisfaction scores, over all simulations, for each bidding type, represented by a different color.
        Args:
            export_results (bool): Whether to export the results to a .csv file
            filter_outliers (bool): Whether to filter out outliers from the results
        """
        all_static_low_bidding_results = []
        all_static_high_bidding_results = []
        all_random_bidding_results = []
        all_free_rider_bidding_results = []
        all_RL_bidding_results = []

        for result_dict in self.all_simulations_satisfaction_scores:
            for epoch in result_dict:
                for (car_copy, score) in result_dict[epoch]:
                    bidding_type = car_copy.bidding_type
                    # Static bidding
                    if bidding_type == 'static_low':
                        all_static_low_bidding_results.append(score)
                    if bidding_type == 'static_high':
                        all_static_high_bidding_results.append(score)
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
            if len(all_static_low_bidding_results) > 0:
                all_static_low_bidding_results = utils.remove_outliers(
                    all_static_low_bidding_results)
            if len(all_static_high_bidding_results) > 0:
                all_static_high_bidding_results = utils.remove_outliers(
                    all_static_high_bidding_results)
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
        if len(all_static_low_bidding_results) > 0:
            plt.hist(all_static_low_bidding_results, bins=30,
                     alpha=0.5, label='Static low bidding')
        if len(all_static_high_bidding_results) > 0:
            plt.hist(all_static_high_bidding_results, bins=30,
                     alpha=0.5, label='Static high bidding')
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

        if export_results == True:
            with open(self.args.results_folder + "/satisfaction_scores_by_type.csv", "w+") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ['Static low bidding', 'Static high bidding' 'Random bidding', 'Free-rider bidding', 'RL bidding'])
                for values in zip_longest(*[all_static_low_bidding_results, all_static_high_bidding_results, all_random_bidding_results, all_free_rider_bidding_results, all_RL_bidding_results]):
                    writer.writerow(values)

    def plot_congestion_heatmap_average(self, export_results=True):
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

        if export_results == True:
            np.savetxt(self.args.results_folder + '/average_congestion_per_intersection.csv',
                       average_congestion_per_intersection, delimiter=",")

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
    """

    def __init__(self, args, grid):
        """ Initialize the MetricsKeeper object
        Args:
            args (argparse.Namespace): Arguments parsed from the command line
            grid (Grid): The grid object of the simulation
        """
        self.args = args
        self.grid = grid
        self.current_sim_satisfaction_scores = {}
        self.total_population_per_intersection = np.zeros(
            (args.grid_size, args.grid_size))

        self.throughput_history_per_intersection = np.zeros(
            (args.grid_size, args.grid_size, args.num_of_epochs))

        self.reward_history_per_intersection = np.zeros(
            (args.grid_size, args.grid_size, args.num_of_epochs))

        self.max_time_waited_history_per_intersection = np.zeros(
            (args.grid_size, args.grid_size, args.num_of_epochs))

        # If discretisation is adaptive_auction_discretization and there are 2 parameters,
        # then there are adaptive_auction_discretization^2 possible combinations of parameters
        self.auction_parameters_space = np.zeros(
            (pow(args.adaptive_auction_discretization, NUM_OF_ADAPT_PARAMS), NUM_OF_ADAPT_PARAMS))

        self.auction_parameters_valuations_per_intersection = [
            [[] for _ in range(args.grid_size)] for _ in range(args.grid_size)]
        self.auction_parameters_counts_per_intersection = [
            [[] for _ in range(args.grid_size)] for _ in range(args.grid_size)]

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
            # Gather the reward history of each intersection
            self.reward_history_per_intersection[x_cord][y_cord] = intersection.get_auction_reward_history(
            )
            # Gather the throughput history of each intersection
            self.throughput_history_per_intersection[x_cord][y_cord] = intersection.get_auction_throughput_history(
            )
            # Gather the max_time_waited history of each intersection
            self.max_time_waited_history_per_intersection[x_cord][y_cord] = intersection.get_max_time_waited_history(
            )

            # Gather the auction parameters and their valuations of each intersection. The parameter space is the same for all intersections
            self.auction_parameters_space, self.auction_parameters_valuations_per_intersection[x_cord][y_cord], self.auction_parameters_counts_per_intersection[x_cord][y_cord] = intersection.get_auction_parameters_and_valuations_and_counts(
            )
