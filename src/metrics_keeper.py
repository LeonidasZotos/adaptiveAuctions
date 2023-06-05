"""A class to keep track of the metrics of the simulation"""
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from math import nan

from src.intersection import Intersection


class MetricsKeeper:
    """
    The MetricsKeeper class is responsible for keeping track of all the evaluation metrics of the simulation, creating graphs etc.
    Attributes:
        all_simulations_results (list): A list of dictionaries, each dictionary containing the satisfaction scores of all cars
        current_sim_satisfaction_scores (dict): A dictionary of satisfaction scores for each car. The key is the epoch, the value is
            a list of all the satisfaction scores for that epoch, if any.
        total_throughput_per_intersection (dict): A dictionary of the total throughput of each intersection. The key is the intersection id,
            the value is the total throughput.
    Functions:
        add_satisfaction_scores: Adds a satisfaction score (and its car id, as a tuple) to the satisfaction_scores dictionary
        produce_results: Produces all the evaluation results of all simulations
        plot_satisfaction_scores_overall_average: Creates a graph of the average satisfaction score per epoch, with error bars, averaged over all simulations
        plot_satisfaction_scores_by_bidding_type: Creates a graph of the average satisfaction score per epoch, with error bars, averaged over all simulations,
            for each bidding type, represented by a different color.
        plot_throughput_heatmap_average: Creates a heatmap of the average throughput per intersection, over all simulations
        ready_for_new_epoch: Prepares the metrics keeper for the next epoch.
        prep_for_new_simulation: Prepares the metrics keeper for a new simulation, by clearing the results of the current simulation
    """

    def __init__(self, args):
        """ Initialize the MetricsKeeper object
        Args:
            all_simulations_results (list): A list of dictionaries, each dictionary containing the satisfaction scores of all cars
                that completed their trip in an epoch, for a single simulation
            current_sim_satisfaction_scores (dict): A dictionary of satisfaction scores for the current simulation. The key is the epoch,
                the value is a list of all the satisfaction scores for that epoch, if any.
        """
        self.all_simulations_results = []
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
        if satisfaction_scores != []:
            self.current_sim_satisfaction_scores[epoch] = satisfaction_scores

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
            return [score for (car_copy, score) in dict]

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

    def plot_satisfaction_scores_by_bidding_type(self, results_folder):
        """Creates a graph of the average satisfaction score per epoch, with error bars, averaged over all simulations,
            for each bidding type, represented by a different color.
            'ohhh 100 lines of code, that's a lot of code (but here we are)'
        Args:
            results_folder (str): The folder in which the results will be stored
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

        plt.xlabel('Epoch')
        plt.ylabel('Average Satisfaction Score \n (the lower, the better)')
        plt.title('Average Satisfaction Score per Epoch')
        plt.legend()
        plt.savefig(results_folder +
                    '/average_satisfaction_score_by_bidding_type.png')
        plt.clf()

    def plot_throughput_heatmap_average(self, results_folder, num_of_simulations):
        """Creates a heatmap of the average throughput per intersection, over all simulations
        Args:
            results_folder (str): The folder in which the results will be stored
            num_of_simulations (int): The number of simulations that were run
        """
        # Create heatmap of average throughput per intersection
        average_throughput_per_intersection = np.floor_divide(
            self.total_throughput_per_intersection, num_of_simulations)  # Divide by number of simulations

        ax = sns.heatmap(average_throughput_per_intersection, annot=True)
        ax.set(xlabel='X coordinate', ylabel='Y coordinate',
               title='Average throughput per intersection')
        plt.savefig(results_folder + '/average_throughput_heatmap.png')
        plt.clf()

    def ready_for_new_epoch(self):
        """Prepares the metrics keeper for the next epoch"""
        # We use a 2d array of the throughput per intersection. The first index is the x coordinate, the second is the y coordinate.
        for intersection in Intersection.all_intersections:
            id = intersection.id
            x_cord = int(id[0])
            y_cord = int(id[1])
            self.total_throughput_per_intersection[x_cord][y_cord] += intersection.num_of_cars_in_intersection()
            self.last_reward_per_intersection[x_cord][y_cord] = intersection.get_last_reward(
            )

    def ready_for_new_simulation(self):
        """Prepares the metrics keeper for a new simulation, by clearing the results of the current simulation"""
        self.all_simulations_results.append(
            self.current_sim_satisfaction_scores)
        self.current_sim_satisfaction_scores = {}
