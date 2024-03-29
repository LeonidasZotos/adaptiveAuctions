""""This file contains the simulations loop. It is responsible for running the simulations and producing
the results. It is called from the aatc.py file.
"""

import os
import sys
import time
from tqdm import tqdm
from multiprocessing import Pool

from src.metrics import MasterKeeper
from src.simulator import Simulator


def run_simulation(args_and_id):
    """Runs a single simulation
    Args:
        args_and_id (tuple): Tuple containing the arguments and the simulation id. Has to be a tuple because of the multiprocessing library
    Returns:
        ResultsKeeper: The results keeper object that contains the results of the simulation
    """
    args, simulation_id = args_and_id
    simulation = Simulator(args, simulation_id)
    results_keeper = simulation.run_simulation()
    return results_keeper


def run(args):
    """Main program that runs the simulation
    Args:
        args (argparse.Namespace): Arguments parsed from the command line
    """
    # Measure execution time
    start_time = time.time()
    sys.setrecursionlimit(25000)

    # Create results folder if it doesn't exist
    if not os.path.exists(args.results_folder):
        os.makedirs(args.results_folder)

    num_of_simulations = args.num_of_simulations
    master_metrics_keeper = MasterKeeper(args)

    pool = Pool()  # Default number of processes will be used

    args_and_ids = [(args, simulation_id)
                    for simulation_id in range(num_of_simulations)]

    print("Number of Processing cores available: ", pool._processes)
    show_progress_bar = True
    if show_progress_bar:
        with tqdm(total=num_of_simulations) as pbar:
            for results_keeper in pool.imap(run_simulation, args_and_ids):
                master_metrics_keeper.store_simulation_results(results_keeper)
                pbar.update()
    else:
        for results_keeper in pool.imap(run_simulation, args_and_ids):
            master_metrics_keeper.store_simulation_results(results_keeper)

    pool.close()
    pool.join()

    # Produce Results
    master_metrics_keeper.produce_results()

    # Print execution time
    print("--- %s seconds ---" % round((time.time() - start_time), 2))
