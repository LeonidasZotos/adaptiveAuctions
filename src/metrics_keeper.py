"""A class to keep track of the metrics of the simulation"""
import matplotlib.pyplot as plt
import numpy as np


class MetricsKeeper:
    """
    The MetricsKeeper class is responsible for keeping track of all the evaluation metrics of the simulation, creating graphs etc.
    Attributes:
        current_sim_satisfaction_scores (dict): A dictionary of satisfaction scores for each car. The key is the epoch, the value is
            a list of all the satisfaction scores for that epoch, if any.
            # TODO: Add attributes
    Functions:
        add_satisfaction_scores: Adds a satisfaction score to the satisfaction_scores dictionary
        # TODO: Add functions
    """

    def __init__(self):
        """ Initialize the MetricsKeeper object
        Args:
            TODO: Add arguments
        """
        self.all_simulations_results = []
        self.current_sim_satisfaction_scores = {}

    def add_satisfaction_scores(self, epoch, satisfaction_scores):
        """Adds the satisfaction scores of the cars that completed a trip. Epochs in which no cars completed a trip receive a None value.
        Args:
            epoch (int): The epoch in which the cars completed their trip
            satisfaction_scores (list): A list of satisfaction scores of the cars that completed their trip
        """
        if satisfaction_scores != []:
            self.current_sim_satisfaction_scores[epoch] = satisfaction_scores

    def produce_results(self, results_folder):
        """Produces all the results of all simulations"""
        # Create a graph of all results.
        self.plot_satisfaction_scores(results_folder)

    def plot_satisfaction_scores(self, results_folder):
        """Creates a graph of the average satisfaction score per epoch, with error bars
        Args:
            results_folder (str): The folder in which the results will be stored
        """
        all_results_dict = {}
        # First, combine all dictionaries into one dictionary
        for result_dict in self.all_simulations_results:
            for epoch in result_dict:
                if epoch in all_results_dict:
                    all_results_dict[epoch] += result_dict[epoch]
                    # print("added: ", result_dict[epoch])
                else:
                    all_results_dict[epoch] = result_dict[epoch]
                    # print("added: ", result_dict[epoch])

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
        plt.ylabel('Average Satisfaction Score')
        plt.title('Average Satisfaction Score per Epoch')
        plt.savefig(results_folder + '/average_satisfaction_score.png')
        plt.clf()

    def prep_for_new_simulation(self):
        """Prepares the metrics keeper for a new simulation"""
        self.all_simulations_results.append(
            self.current_sim_satisfaction_scores)
        self.current_sim_satisfaction_scores = {}
