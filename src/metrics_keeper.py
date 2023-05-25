"""A class to keep track of the metrics of the simulation"""
import matplotlib.pyplot as plt
import numpy as np
from math import nan

import src.utils as utils


class MetricsKeeper:
    """
    The MetricsKeeper class is responsible for keeping track of all the evaluation metrics of the simulation, creating graphs etc.
    Attributes:
        all_simulations_results (list): A list of dictionaries, each dictionary containing the satisfaction scores of all cars
        current_sim_satisfaction_scores (dict): A dictionary of satisfaction scores for each car. The key is the epoch, the value is
            a list of all the satisfaction scores for that epoch, if any.
    Functions:
        add_satisfaction_scores: Adds a satisfaction score (and its car id, as a tuple) to the satisfaction_scores dictionary
        produce_results: Produces all the evaluation results of all simulations
        plot_satisfaction_scores_overall_average: Creates a graph of the average satisfaction score per epoch, with error bars, averaged over all simulations
        prep_for_new_simulation: Prepares the metrics keeper for a new simulation, by clearing the results of the current simulation
    """

    def __init__(self):
        """ Initialize the MetricsKeeper object
        Args:
            all_simulations_results (list): A list of dictionaries, each dictionary containing the satisfaction scores of all cars
                that completed their trip in an epoch, for a single simulation
            current_sim_satisfaction_scores (dict): A dictionary of satisfaction scores for the current simulation. The key is the epoch,
                the value is a list of all the satisfaction scores for that epoch, if any.
        """
        self.all_simulations_results = []
        self.current_sim_satisfaction_scores = {}

    def add_satisfaction_scores(self, epoch, satisfaction_scores):
        """Adds the satisfaction scores of the cars that completed a trip. If there was no car that completed 
            a trip in an epoch, there is no entry for that epoch.
        Args:
            epoch (int): The epoch in which the cars completed their trip
            satisfaction_scores (list): A list of tuples, containing car ids and their satisfaction scores of the completed trip
        """
        if satisfaction_scores != []:
            self.current_sim_satisfaction_scores[epoch] = satisfaction_scores

    def produce_results(self, results_folder):
        """Produces all the evaluation results of all simulations
        Args:
            results_folder (str): The folder in which the results will be stored
        """
        # Create a graph of all satisfaction scores
        self.plot_satisfaction_scores_overall_average(results_folder)

        # Create a graph of all satisfaction scores, per bidding type
        self.plot_satisfaction_scores_by_bidding_type(results_folder)

    def plot_satisfaction_scores_overall_average(self, results_folder):
        """Creates a graph of the average satisfaction score per epoch, with error bars, averaged over all simulations.
        Args:
            results_folder (str): The folder in which the results will be stored
        """

        def remove_car_ids_from_dict(dict):
            """Removes the car ids from the dictionary, so that it only contains the satisfaction scores"""
            return [score for (id, score) in dict]

        all_results_dict = {}
        # First, combine all dictionaries into one dictionary
        for result_dict in self.all_simulations_results:
            for epoch in result_dict:
                if epoch in all_results_dict:
                    all_results_dict[epoch] += remove_car_ids_from_dict(
                        result_dict[epoch])
                else:
                    all_results_dict[epoch] = remove_car_ids_from_dict(
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
        """

        def get_bidding_type_from_id(car_id):
            """Retrieves the bidding type of a car, given its id"""
            return utils.get_car(car_id).bidding_type

        static_bidding_results = {}
        random_bidding_results = {}
        RL_bidder_results = {}
        
        
        
        
        for result_dict in self.all_simulations_results:
            for epoch in result_dict:
                for (car_id, score) in result_dict[epoch]:
                    bidding_type = get_bidding_type_from_id(car_id)
                    if bidding_type == 'static':
                        if epoch in static_bidding_results:
                            static_bidding_results[epoch].append(score)
                        else:
                            static_bidding_results[epoch] = [score]
                    elif bidding_type == 'random':
                        if epoch in random_bidding_results:
                            random_bidding_results[epoch].append(score)
                        else:
                            random_bidding_results[epoch] = [score]
                    elif bidding_type == 'RL':
                        if epoch in RL_bidder_results:
                            RL_bidder_results[epoch].append(score)
                        else:
                            RL_bidder_results[epoch] = [score]



        
        # print avereage of each bidding type
        print("static bidding average: ", sum([sum(static_bidding_results[epoch]) / len(static_bidding_results[epoch]) for epoch in static_bidding_results]) / len(static_bidding_results))
        print("random bidding average: ", sum([sum(random_bidding_results[epoch]) / len(random_bidding_results[epoch]) for epoch in random_bidding_results]) / len(random_bidding_results))
        print("RL bidding average: ", sum([sum(RL_bidder_results[epoch]) / len(RL_bidder_results[epoch]) for epoch in RL_bidder_results]) / len(RL_bidder_results))
        
        # Create a list of all epochs in which cars completed their trip
        epochs = []
        for epoch in static_bidding_results:
            if static_bidding_results[epoch] != None or random_bidding_results[epoch] != None or RL_bidder_results[epoch] != None:
                epochs.append(epoch)

        # Create a list of all average satisfaction scores
        static_bidding_average_satisfaction_scores = []
        random_bidding_average_satisfaction_scores = []
        RL_bidder_average_satisfaction_scores = []
        for epoch in epochs:
            if epoch in static_bidding_results:
                static_bidding_average_satisfaction_scores.append(
                    sum(static_bidding_results[epoch]) / len(static_bidding_results[epoch]))
            else:
                static_bidding_average_satisfaction_scores.append(nan)
            if epoch in random_bidding_results:
                random_bidding_average_satisfaction_scores.append(
                    sum(random_bidding_results[epoch]) / len(random_bidding_results[epoch]))
            else:
                random_bidding_average_satisfaction_scores.append(nan)
            if epoch in RL_bidder_results:
                RL_bidder_average_satisfaction_scores.append(
                    sum(RL_bidder_results[epoch]) / len(RL_bidder_results[epoch]))
            else:
                RL_bidder_average_satisfaction_scores.append(nan)

        # Create a list of all standard deviations of satisfaction scores
        static_bidding_standard_deviations = []
        random_bidding_standard_deviations = []
        RL_bidder_standard_deviations = []
        for epoch in epochs:
            if epoch in static_bidding_results:
                static_bidding_standard_deviations.append(
                    np.std(static_bidding_results[epoch]))
            else:
                static_bidding_standard_deviations.append(nan)
            if epoch in random_bidding_results:
                random_bidding_standard_deviations.append(
                    np.std(random_bidding_results[epoch]))
            else:
                random_bidding_standard_deviations.append(nan)
            if epoch in RL_bidder_results:
                RL_bidder_standard_deviations.append(
                    np.std(RL_bidder_results[epoch]))
            else:
                RL_bidder_standard_deviations.append(nan)

        # Plot the average satisfaction score per epoch, per bidding type & with error bars
        plt.errorbar(epochs, static_bidding_average_satisfaction_scores,
                     yerr=static_bidding_standard_deviations, fmt='o', label='Static bidding')
        plt.errorbar(epochs, random_bidding_average_satisfaction_scores,
                     yerr=random_bidding_standard_deviations, fmt='o', label='Random bidding')
        plt.errorbar(epochs, RL_bidder_average_satisfaction_scores,
                     yerr=RL_bidder_standard_deviations, fmt='o', label='RL bidding')

        plt.xlabel('Epoch')
        plt.ylabel('Average Satisfaction Score \n (the lower, the better)')
        plt.title('Average Satisfaction Score per Epoch')
        plt.legend()
        plt.savefig(results_folder +
                    '/average_satisfaction_score_by_bidding_type.png')
        plt.clf()

    def prep_for_new_simulation(self):
        """Prepares the metrics keeper for a new simulation, by clearing the results of the current simulation"""
        self.all_simulations_results.append(
            self.current_sim_satisfaction_scores)
        self.current_sim_satisfaction_scores = {}
