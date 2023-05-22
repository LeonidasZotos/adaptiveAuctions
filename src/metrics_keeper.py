"""A class to keep track of the metrics of the simulation"""
import matplotlib.pyplot as plt
import numpy as np


class MetricsKeeper:
    """
    The MetricsKeeper class is responsible for keeping track of all the evaluation metrics of the simulation, creating graphs etc.
    Attributes:
        all_simulations_results (list): A list of dictionaries, each dictionary containing the satisfaction scores of all cars
        current_sim_satisfaction_scores (dict): A dictionary of satisfaction scores for each car. The key is the epoch, the value is
            a list of all the satisfaction scores for that epoch, if any.
    Functions:
        add_satisfaction_scores: Adds a satisfaction score to the satisfaction_scores dictionary
        produce_results: Produces all the evaluation results of all simulations
        plot_satisfaction_scores: Creates a graph of the average satisfaction score per epoch, with error bars, averaged over all simulations
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
            satisfaction_scores (list): A list of satisfaction scores of the cars that completed their trip
        """
        if satisfaction_scores != []:
            self.current_sim_satisfaction_scores[epoch] = satisfaction_scores

    def produce_results(self, results_folder):
        """Produces all the evaluation results of all simulations
        Args:
            results_folder (str): The folder in which the results will be stored
        """
        # Create a graph of all satisfaction scores
        self.plot_satisfaction_scores(results_folder)

    def plot_satisfaction_scores(self, results_folder):
        """Creates a graph of the average satisfaction score per epoch, with error bars, averaged over all simulations.
        Args:
            results_folder (str): The folder in which the results will be stored
        """
        all_results_dict = {}
        # First, combine all dictionaries into one dictionary
        for result_dict in self.all_simulations_results:
            for epoch in result_dict:
                if epoch in all_results_dict:
                    all_results_dict[epoch] += result_dict[epoch]
                else:
                    all_results_dict[epoch] = result_dict[epoch]

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

    def prep_for_new_simulation(self):
        """Prepares the metrics keeper for a new simulation, by clearing the results of the current simulation"""
        self.all_simulations_results.append(
            self.current_sim_satisfaction_scores)
        self.current_sim_satisfaction_scores = {}
