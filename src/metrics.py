"""A class to keep track of the metrics of the simulation"""
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from math import nan
import pandas as pd
import csv
from itertools import zip_longest

import src.utils as utils


class MasterKeeper:
    """
    The MasterKeeper class is responsible for keeping track of all the metrics for all simulations.
    Attributes:
        all_simulations_results (list): A list of dictionaries, each dictionary containing the satisfaction scores of all cars
            that completed their trip in an epoch, for a single simulation
        total_throughput_per_intersection (dict): A dictionary of the total throughput of each intersection. The key is the intersection id,
            the value is the total throughput.
    Functions:
        store_simulation_results(sim_metrics_keeper): Stores the results of a single simulation
        produce_results(): Produces all the evaluation results of all simulations
        plot_satisfaction_scores_overall_average(): Creates a graph of the average satisfaction score per epoch, with error bars, averaged over all simulations
        plot_satisfaction_scores_by_bidding_type(, with_std=False, export_results=True, filter_outliers=True): Creates a 
            graph of the average satisfaction score per epoch, with error bars, averaged over all simulations,
            for each bidding type, represented by a different color.
        histogram_satisfaction_scores_by_bidding_type(, export_results=True, filter_outliers=True): Creates a histogram of all satisfaction scores,
            over all simulations, per bidding type
        plot_throughput_heatmap_average(, export_results=True): Creates a heatmap of the 
            average tƒhroughput per intersection, over all simulations    
        plot_throughput_per_intersection_history(export_results=True): Creates a plot with subplots for each intersection.
            Each subplot is a graph of the throughput history of that intersection. In total there are as many subplots as intersections
        plot_reward_per_intersection_history(export_results=True): Creates a plot with subplots for each intersection.
            Each subplot is a graph of the reward history of that intersection. In total there are as many subplots as intersections
        plot_adaptive_auction_parameters_valuations_per_intersection(): Creates a plot with subplots for each intersection.
            Each subplot is a graph of the valuations of the auction parameters of that intersection.
    """

    def __init__(self, args):
        """ Initialize the MetricsKeeper object
        Args:
            args (argparse.Namespace): Arguments parsed from the command line
        """
        self.args = args
        self.all_simulations_satisfaction_scores = []

        # Total throughput per intersection
        self.total_throughput_per_intersection = np.zeros(
            (self.args.grid_size, self.args.grid_size))

        # Total throughput history per intersection
        self.total_throughput_history_per_intersection = np.zeros(
            (self.args.grid_size, self.args.grid_size, self.args.num_of_epochs))

        # Number of measurements per intersection, used to calculate the average
        self.count_of_measurements_per_intersection = np.zeros(
            (self.args.grid_size, self.args.grid_size, self.args.num_of_epochs))

        # Total reward history per intersection
        self.total_reward_history_per_intersection = np.zeros(
            (self.args.grid_size, self.args.grid_size, self.args.num_of_epochs))

        # Auction parameter space
        self.auction_parameters_space = []

        # Sum of auction parameters valuations per intersection
        self.sum_auction_parameters_valuations_per_intersection = np.zeros(
            (self.args.grid_size, self.args.grid_size, pow(self.args.adaptive_auction_discretization, 2)))

    def store_simulation_results(self, sim_metrics_keeper):
        """Prepares the metrics keeper for a new simulation, by clearing the results of the current simulation
        Args: 
            sim_metrics_keeper (SimulationMetrics): The metrics keeper of the current simulation
        """
        self.all_simulations_satisfaction_scores.append(
            sim_metrics_keeper.current_sim_satisfaction_scores)

        self.total_throughput_per_intersection += sim_metrics_keeper.total_throughput_per_intersection

        self.total_throughput_history_per_intersection += sim_metrics_keeper.throughput_history_per_intersection

        # For each measurement that is not nan, we add 1 to the count of measurements, so that we can later calculate the average
        self.count_of_measurements_per_intersection += np.where(
            sim_metrics_keeper.reward_history_per_intersection != nan, 1, 0)
        self.total_reward_history_per_intersection += np.nan_to_num(
            sim_metrics_keeper.reward_history_per_intersection)

        if self.args.auction_modifier_type == "bandit":  # Only relevant for adaptive auctions
            # Retrieve the parameter space and the valuations per intersection. The parameters space is the same for all intersections
            self.auction_parameters_space = sim_metrics_keeper.auction_parameters_space

            self.sum_auction_parameters_valuations_per_intersection += sim_metrics_keeper.auction_parameters_valuations_per_intersection

    def produce_results(self):
        """Produces all the evaluation results of all simulations"""
        # Create a .txt file with the arguments used for the simulation
        with open(self.args.results_folder + '/configuration.txt', 'w') as f:
            for arg in vars(self.args):
                f.write(arg + ': ' + str(getattr(self.args, arg)) + '\n')

        # Create a graph of all satisfaction scores, over all simulations
        self.plot_satisfaction_scores_overall_average()

        # Create a graph of all satisfaction scores, per bidding type, over all simulations
        self.plot_satisfaction_scores_by_bidding_type()

        # Create a histogram of all satisfaction scores, over all simulations, per bidding type
        self.histogram_satisfaction_scores_by_bidding_type()

        # Create a heatmap of the average throughput per intersection, over all simulations
        self.plot_throughput_heatmap_average()

        self.plot_throughput_per_intersection_history()

        # Create a graph with graphs of the average reward per intersection, over all simulations
        self.plot_reward_per_intersection_history()

        if self.args.auction_modifier_type == "bandit":
            self.plot_adaptive_auction_parameters_valuations_per_intersection()

    def plot_satisfaction_scores_overall_average(self):
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

        # Plot the average satisfaction score per epoch, with error bars
        plt.errorbar(epochs, average_satisfaction_scores,
                     yerr=standard_deviations, fmt='o')
        plt.xlabel('Epoch')
        plt.ylabel('Average Satisfaction Score \n (the higher, the better)')
        plt.title('Average Satisfaction Score per Epoch')
        plt.savefig(self.args.results_folder +
                    '/average_satisfaction_score.png')
        plt.clf()

    def plot_satisfaction_scores_by_bidding_type(self, with_std=False, export_results=True, filter_outliers=True):
        """Creates a graph of the average satisfaction score per epoch, with error bars, averaged over all simulations,
            for each bidding type, represented by a different color.
            'ohhh almost 200 lines of code, that's a lot of code for just one function (but here we are)'
        Args:
            with_std (bool): Whether to include the standard deviation in the plot
            export_results (bool): Whether to export the results to a .csv file
            filter_outliers (bool): Whether to filter out outliers from the results
        """

        static_bidding_results = {}
        random_bidding_results = {}
        free_rider_bidding_results = {}
        RL_bidding_results = {}

        for result_dict in self.all_simulations_satisfaction_scores:
            for epoch in result_dict:
                for (car_copy, score) in result_dict[epoch]:
                    bidding_type = car_copy.bidding_type
                    # Static bidding
                    if bidding_type == 'static':
                        if epoch in static_bidding_results:
                            static_bidding_results[epoch].append(score)
                        else:
                            static_bidding_results[epoch] = [score]
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
        for epoch in static_bidding_results:
            if static_bidding_results[epoch] != None or random_bidding_results[epoch] != None or free_rider_bidding_results[epoch] != None or RL_bidding_results[epoch] != None:
                epochs.append(epoch)

        # Remove outliers if necessary:
        if filter_outliers == True:
            for epoch in epochs:
                # Static bidding:
                if epoch in static_bidding_results:
                    static_bidding_results[epoch] = utils.remove_outliers(
                        static_bidding_results[epoch])
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
        static_bidding_average_satisfaction_scores = []
        random_bidding_average_satisfaction_scores = []
        free_rider_bidding_average_satisfaction_scores = []
        RL_bidder_average_satisfaction_scores = []

        for epoch in epochs:
            # Static bidding:
            if epoch in static_bidding_results:
                static_bidding_average_satisfaction_scores.append(
                    sum(static_bidding_results[epoch]) / len(static_bidding_results[epoch]))
            else:
                static_bidding_average_satisfaction_scores.append(nan)
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
        static_bidding_sd = []
        random_bidding_sd = []
        free_rider_bidding_sd = []
        RL_bidder_sd = []
        for epoch in epochs:
            # Static bidding:
            if epoch in static_bidding_results:
                static_bidding_sd.append(
                    np.std(static_bidding_results[epoch]))
            else:
                static_bidding_sd.append(nan)
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
        if with_std == True:
            # Plot the average satisfaction score per epoch, per bidding type & with error bars
            if len(static_bidding_results) > 0:
                plt.errorbar(epochs, static_bidding_average_satisfaction_scores,
                             yerr=static_bidding_sd, fmt='o', label='Static bidding')
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
            if len(static_bidding_results) > 0:
                plt.plot(
                    epochs, static_bidding_average_satisfaction_scores, 'o', linestyle='None', label='Static bidding', markersize=1.5)
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
                epochs, static_bidding_average_satisfaction_scores, static_bidding_sd,
                random_bidding_average_satisfaction_scores, random_bidding_sd,
                free_rider_bidding_average_satisfaction_scores, free_rider_bidding_sd,
                RL_bidder_average_satisfaction_scores, RL_bidder_sd
            ]).T, delimiter=",", header="Epoch, Static bidding Score, Static bidding SD, Random bidding Score, Random bidding SD, Free-rider bidding Score, Free-rider bidding SD, RL bidding Score, RL bidding SD")

    def histogram_satisfaction_scores_by_bidding_type(self, export_results=True, filter_outliers=True):
        """Creates a histogram of all satisfaction scores, over all simulations, for each bidding type, represented by a different color.
        Args:
            export_results (bool): Whether to export the results to a .csv file
            filter_outliers (bool): Whether to filter out outliers from the results
        """
        all_static_bidding_results = []
        all_random_bidding_results = []
        all_free_rider_bidding_results = []
        all_RL_bidding_results = []

        for result_dict in self.all_simulations_satisfaction_scores:
            for epoch in result_dict:
                for (car_copy, score) in result_dict[epoch]:
                    bidding_type = car_copy.bidding_type
                    # Static bidding
                    if bidding_type == 'static':
                        all_static_bidding_results.append(score)
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
            if len(all_static_bidding_results) > 0:
                all_static_bidding_results = utils.remove_outliers(
                    all_static_bidding_results)
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
        if len(all_static_bidding_results) > 0:
            plt.hist(all_static_bidding_results, bins=30,
                     alpha=0.5, label='Static bidding')
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
        plt.ylabel('Frequency')
        plt.title('Histogram of Satisfaction Scores')
        plt.legend()
        plt.savefig(self.args.results_folder +
                    '/histogram_satisfaction_scores_by_bidding_type.png')
        plt.clf()

        if export_results == True:
            with open(self.args.results_folder + "/satisfaction_scores_by_type.csv", "w+") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ['Static bidding', 'Random bidding', 'Free-rider bidding', 'RL bidding'])
                for values in zip_longest(*[all_static_bidding_results, all_random_bidding_results, all_free_rider_bidding_results, all_RL_bidding_results]):
                    writer.writerow(values)

    def plot_throughput_heatmap_average(self, export_results=True):
        """Creates a heatmap of the average throughput per intersection, over all simulations
        Args:
            export_results (bool): Whether to export the results to a .csv file
        """
        # Create heatmap of average throughput per intersection
        average_throughput_per_intersection = np.floor_divide(
            self.total_throughput_per_intersection, self.args.num_of_simulations)  # Divide by number of simulations

        ax = sns.heatmap(average_throughput_per_intersection, annot=True)
        ax.set(xlabel='X coordinate', ylabel='Y coordinate',
               title='Average throughput per intersection')
        plt.savefig(self.args.results_folder +
                    '/average_throughput_heatmap.png')
        plt.clf()

        if export_results == True:
            np.savetxt(self.args.results_folder + '/average_throughput_per_intersection.csv',
                       average_throughput_per_intersection, delimiter=",")

    def plot_throughput_per_intersection_history(self, export_results=True):
        # Divide by the number of measurements per intersection to calculate the average. If there are no measurements, the average is 0
        average_throughput_per_intersection = np.divide(
            self.total_throughput_history_per_intersection, self.args.num_of_simulations)
        # Create a plot with subplots for each intersection. Each subplot is a graph of the reward history of that intersection. In total there are as many subplots as intersections
        fig, axs = plt.subplots(
            self.total_throughput_history_per_intersection.shape[0], self.total_throughput_history_per_intersection.shape[1], sharex=True, sharey=True, figsize=(20, 20))
        for i in range(self.total_throughput_history_per_intersection.shape[0]):
            for j in range(self.total_throughput_history_per_intersection.shape[1]):
                axs[i, j].plot(average_throughput_per_intersection[i, j])
                axs[i, j].set_title('[' + str(i) + str(j) + ']')
                axs[i, j].set_xlabel('Epoch')
                axs[i, j].set_ylabel('Average Throughput')
        plt.savefig(self.args.results_folder +
                    '/average_throughput_per_intersection_history.png')
        plt.clf()

        if export_results == True:
            # Create a pandas dataframe, where each intersection is a column. The column header is the coordinates of the intersection
            rewards_history_df = pd.DataFrame()
            for i in range(self.total_reward_history_per_intersection.shape[0]):
                for j in range(self.total_reward_history_per_intersection.shape[1]):
                    rewards_history_df[str(
                        i) + '_' + str(j)] = average_throughput_per_intersection[i, j]
            rewards_history_df.to_csv(self.args.results_folder +
                                      '/average_throughput_per_intersection_history.csv', index=False)

    def plot_reward_per_intersection_history(self, export_results=True):
        # Divide by the number of measurements per intersection to calculate the average. If there are no measurements, the average is 0
        average_reward_per_intersection = np.divide(
            self.total_reward_history_per_intersection, self.count_of_measurements_per_intersection)
        # Create a plot with subplots for each intersection. Each subplot is a graph of the reward history of that intersection. In total there are as many subplots as intersections
        fig, axs = plt.subplots(
            self.total_reward_history_per_intersection.shape[0], self.total_reward_history_per_intersection.shape[1], sharex=True, sharey=True, figsize=(20, 20))
        for i in range(self.total_reward_history_per_intersection.shape[0]):
            for j in range(self.total_reward_history_per_intersection.shape[1]):
                axs[i, j].plot(average_reward_per_intersection[i, j])
                axs[i, j].set_title('[' + str(i) + str(j) + ']')
                axs[i, j].set_xlabel('Epoch')
                axs[i, j].set_ylabel('Average Reward')
        plt.savefig(self.args.results_folder +
                    '/average_reward_per_intersection_history.png')
        plt.clf()

        if export_results == True:
            # Create a pandas dataframe, where each intersection is a column. The column header is the coordinates of the intersection
            rewards_history_df = pd.DataFrame()
            for i in range(self.total_reward_history_per_intersection.shape[0]):
                for j in range(self.total_reward_history_per_intersection.shape[1]):
                    rewards_history_df[str(
                        i) + '_' + str(j)] = average_reward_per_intersection[i, j]
            rewards_history_df.to_csv(self.args.results_folder +
                                      '/average_reward_per_intersection_history.csv', index=False)

    def plot_adaptive_auction_parameters_valuations_per_intersection(self):
        """Creates a plot of subplots for each intersection. Each subplot is a 3d subplot of the evaluation per parameter set."""
        # Divide by all the valuations for each parameter set by the number of simulations to calculate the average.
        average_reward_per_parameter_set_per_intersection = np.divide(
            self.sum_auction_parameters_valuations_per_intersection, self.args.num_of_simulations)

        parameter_space_2d = np.reshape(
            self.auction_parameters_space, (self.args.adaptive_auction_discretization, self.args.adaptive_auction_discretization, 2))
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
        throughput_history_per_intersection (dict): A dictionary of the throughput history of each intersection. The key is the intersection id,
            the value is the throughput history.
        reward_history_per_intersection (dict): A dictionary of the reward history of each intersection. The key is the intersection id,
            the value is the reward history.
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
        self.total_throughput_per_intersection = np.zeros(
            (args.grid_size, args.grid_size))

        self.throughput_history_per_intersection = np.zeros(
            (args.grid_size, args.grid_size, args.num_of_epochs))

        self.reward_history_per_intersection = np.zeros(
            (args.grid_size, args.grid_size, args.num_of_epochs))

        # If discretisation is adaptive_auction_discretization and there are 2 parameters,
        # then there are adaptive_auction_discretization^2 possible combinations of parameters
        self.auction_parameters_space = np.zeros(
            (pow(args.adaptive_auction_discretization, 2), 2))

        self.auction_parameters_valuations_per_intersection = [
            [[] for _ in range(args.grid_size)] for _ in range(args.grid_size)]

    def add_satisfaction_scores(self, epoch, satisfaction_scores):
        """Adds the satisfaction scores of the cars that completed a trip. If there was no car that completed 
            a trip in an epoch, there is no entry for that epoch.
        Args:
            epoch (int): The epoch in which the cars completed their trip
            satisfaction_scores (list): A list of tuples, containing small car copies and their satisfaction scores of the completed trip
        """
        if satisfaction_scores:  # if it is not empty
            self.current_sim_satisfaction_scores[epoch] = satisfaction_scores

    def ready_for_new_epoch(self):
        """Prepares the metrics keeper for the next epoch"""
        # We use a 2d array. The first index is the x coordinate, the second is the y coordinate.
        # Here, we store all the total throughput per intersection and the last reward per intersection
        for intersection in self.grid.all_intersections:
            id = intersection.id
            x_cord, y_cord = map(int, id)
            self.total_throughput_per_intersection[x_cord][y_cord] += intersection.get_num_of_cars_in_intersection()

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

            if self.args.auction_modifier_type == "bandit":  # Only if adaptive auction is used
                # Gather the auction parameters and their valuations of each intersection. The parameter space is the same for all intersections
                self.auction_parameters_space, self.auction_parameters_valuations_per_intersection[x_cord][y_cord] = intersection.get_auction_parameters_and_valuations(
                )
