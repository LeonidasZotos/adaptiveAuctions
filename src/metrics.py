"""A class to keep track of the metrics of the simulation"""
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from math import nan

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
        produce_results(args): Produces all the evaluation results of all simulations
        plot_satisfaction_scores_overall_average(results_folder): Creates a graph of the average satisfaction score per epoch, with error bars, averaged over all simulations
        plot_satisfaction_scores_by_bidding_type(results_folder, with_std=False, export_results=True, filter_outliers=True): Creates a 
        graph of the average satisfaction score per epoch, with error bars, averaged over all simulations,
            for each bidding type, represented by a different color.
        plot_throughput_heatmap_average(results_folder, num_of_simulations, export_results=True): Creates a heatmap of the 
            average throughput per intersection, over all simulations    
    """

    def __init__(self, args):
        """ Initialize the MetricsKeeper object
        Args:
            args (argparse.Namespace): Arguments parsed from the command line
        """
        self.all_simulations_results = []
        self.total_throughput_per_intersection = np.zeros(
            (args.grid_size, args.grid_size))

    def store_simulation_results(self, sim_metrics_keeper):
        """Prepares the metrics keeper for a new simulation, by clearing the results of the current simulation"""
        self.all_simulations_results.append(
            sim_metrics_keeper.current_sim_satisfaction_scores)
        self.total_throughput_per_intersection += sim_metrics_keeper.total_throughput_per_intersection

    def produce_results(self, args):
        """Produces all the evaluation results of all simulations
        Args:
            args (argparse.Namespace): Arguments parsed from the command line
        """
        # Create a .txt file with the arguments used for the simulation
        with open(args.results_folder + '/configuration.txt', 'w') as f:
            for arg in vars(args):
                f.write(arg + ': ' + str(getattr(args, arg)) + '\n')

        # Create a graph of all satisfaction scores, over all simulations
        self.plot_satisfaction_scores_overall_average(args.results_folder)

        # Create a graph of all satisfaction scores, per bidding type, over all simulations
        self.plot_satisfaction_scores_by_bidding_type(args.results_folder)

        # Create a heatmap of the average throughput per intersection, over all simulations
        self.plot_throughput_heatmap_average(
            args.results_folder, args.num_of_simulations)

    def plot_satisfaction_scores_overall_average(self, results_folder):
        """Creates a graph of the average satisfaction score per epoch, with error bars, averaged over all simulations.
        Args:
            results_folder (str): The folder in which the results will be stored
        """

        def remove_car_copies_from_dict(dict):
            """Removes the car copies from the dictionary, so that it only contains the satisfaction scores"""
            return [score for (_, score) in dict]

        all_results_dict = {}
        # First, combine all dictionaries into one dictionary
        for result_dict in self.all_simulations_results:
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
        plt.ylabel('Average Satisfaction Score \n (the lower, the better)')
        plt.title('Average Satisfaction Score per Epoch')
        plt.savefig(results_folder + '/average_satisfaction_score.png')
        plt.clf()

    def plot_satisfaction_scores_by_bidding_type(self, results_folder, with_std=False, export_results=True, filter_outliers=True):
        """Creates a graph of the average satisfaction score per epoch, with error bars, averaged over all simulations,
            for each bidding type, represented by a different color.
            'ohhh almost 200 lines of code, that's a lot of code for just one function (but here we are)'
        Args:
            results_folder (str): The folder in which the results will be stored
            with_std (bool): Whether to include the standard deviation in the plot
            export_results (bool): Whether to export the results to a .csv file
            filter_outliers (bool): Whether to filter out outliers from the results
        """

        static_bidding_results = {}
        random_bidding_results = {}
        free_rider_bidding_results = {}
        RL_bidding_results = {}

        for result_dict in self.all_simulations_results:
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

        if with_std == True:
            # Create a list of all standard deviations of satisfaction scores
            static_bidding_standard_deviations = []
            random_bidding_standard_deviations = []
            free_rider_bidding_standard_deviations = []
            RL_bidder_standard_deviations = []
            for epoch in epochs:
                # Static bidding:
                if epoch in static_bidding_results:
                    static_bidding_standard_deviations.append(
                        np.std(static_bidding_results[epoch]))
                else:
                    static_bidding_standard_deviations.append(nan)
                # Random bidding:
                if epoch in random_bidding_results:
                    random_bidding_standard_deviations.append(
                        np.std(random_bidding_results[epoch]))
                else:
                    random_bidding_standard_deviations.append(nan)
                # Free-Rider bidding:
                if epoch in free_rider_bidding_results:
                    free_rider_bidding_standard_deviations.append(
                        np.std(free_rider_bidding_results[epoch]))
                else:
                    free_rider_bidding_standard_deviations.append(nan)
                # RL bidding:
                if epoch in RL_bidding_results:
                    RL_bidder_standard_deviations.append(
                        np.std(RL_bidding_results[epoch]))
                else:
                    RL_bidder_standard_deviations.append(nan)

            # Plot the average satisfaction score per epoch, per bidding type & with error bars
            if len(static_bidding_results) > 0:
                plt.errorbar(epochs, static_bidding_average_satisfaction_scores,
                             yerr=static_bidding_standard_deviations, fmt='o', label='Static bidding')
            if len(random_bidding_results) > 0:
                plt.errorbar(epochs, random_bidding_average_satisfaction_scores,
                             yerr=random_bidding_standard_deviations, fmt='o', label='Random bidding')
            if len(free_rider_bidding_results) > 0:
                plt.errorbar(epochs, free_rider_bidding_average_satisfaction_scores,
                             yerr=free_rider_bidding_standard_deviations, fmt='o', label='Free-rider bidding')
            if len(RL_bidding_results) > 0:
                plt.errorbar(epochs, RL_bidder_average_satisfaction_scores,
                             yerr=RL_bidder_standard_deviations, fmt='o', label='RL bidding')
        else:
            # Plot the average satisfaction score per epoch, per bidding type (without error bars)
            if len(static_bidding_results) > 0:
                plt.plot(
                    epochs, static_bidding_average_satisfaction_scores, 'o', linestyle='None', label='Static bidding')
            if len(random_bidding_results) > 0:
                plt.plot(
                    epochs, random_bidding_average_satisfaction_scores, 'o', linestyle='None', label='Random bidding')
            if len(free_rider_bidding_results) > 0:
                plt.plot(epochs, free_rider_bidding_average_satisfaction_scores,
                         'o', linestyle='None', label='Free-rider bidding')
            if len(RL_bidding_results) > 0:
                plt.plot(
                    epochs, RL_bidder_average_satisfaction_scores, 'o', linestyle='None', label='RL bidding')

        plt.xlabel('Epoch')
        plt.ylabel('Average Satisfaction Score \n (the lower, the better)')
        plt.title('Average Satisfaction Score per Epoch')
        plt.legend()
        plt.savefig(results_folder +
                    '/average_satisfaction_score_by_bidding_type.png')
        plt.clf()

        if export_results == True:
            np.savetxt(results_folder + '/average_satisfaction_score_by_bidding_type.csv', np.array([epochs, static_bidding_average_satisfaction_scores, random_bidding_average_satisfaction_scores,
                       free_rider_bidding_average_satisfaction_scores, RL_bidder_average_satisfaction_scores]).T, delimiter=",", header="Epoch, Static bidding, Random bidding, Free-rider bidding, RL bidding")

    def plot_throughput_heatmap_average(self, results_folder, num_of_simulations, export_results=True):
        """Creates a heatmap of the average throughput per intersection, over all simulations
        Args:
            results_folder (str): The folder in which the results will be stored
            num_of_simulations (int): The number of simulations that were run
            export_results (bool): Whether to export the results to a .csv file
        """
        # Create heatmap of average throughput per intersection
        average_throughput_per_intersection = np.floor_divide(
            self.total_throughput_per_intersection, num_of_simulations)  # Divide by number of simulations

        ax = sns.heatmap(average_throughput_per_intersection, annot=True)
        ax.set(xlabel='X coordinate', ylabel='Y coordinate',
               title='Average throughput per intersection')
        plt.savefig(results_folder + '/average_throughput_heatmap.png')
        plt.clf()

        if export_results == True:
            np.savetxt(results_folder + '/average_throughput_per_intersection.csv',
                       average_throughput_per_intersection, delimiter=",")


class SimulationMetrics:
    """
    The SimulationMetrics class is responsible for keeping track of all the metrics for a single simulation.
    Attributes:
        grid (Grid): The grid object of the simulation
        current_sim_satisfaction_scores (dict): A dictionary of satisfaction scores for each car. The key is the epoch, the value is
            a list of all the satisfaction scores for that epoch, if any.
        total_throughput_per_intersection (dict): A dictionary of the total throughput of each intersection. The key is the intersection id,
            the value is the total throughput.
        last_reward_per_intersection (dict): A dictionary of the last reward of each intersection. The key is the intersection id,
            the value is the last reward.
    Functions:
        add_satisfaction_scores(epoch, satisfaction_scores): Adds the satisfaction scores of the cars that completed a trip.
            If there was no car that completed a trip in an epoch, there is no entry for that epoch.
        ready_for_new_epoch(): Prepares the metrics keeper for the next epoch
    """

    def __init__(self, args, grid):
        """ Initialize the MetricsKeeper object
        Args:
            current_sim_satisfaction_scores (dict): A dictionary of satisfaction scores for the current simulation. The key is the epoch,
                the value is a list of all the satisfaction scores for that epoch, if any.
        """
        self.grid = grid
        self.current_sim_satisfaction_scores = {}
        self.total_throughput_per_intersection = np.zeros(
            (args.grid_size, args.grid_size))
        self.last_reward_per_intersection = np.zeros(
            (args.grid_size, args.grid_size))

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
            self.total_throughput_per_intersection[x_cord][y_cord] += intersection.num_of_cars_in_intersection()
            self.last_reward_per_intersection[x_cord][y_cord] = intersection.get_last_reward(
            )
