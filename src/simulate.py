""""This file contains the main simulation loop. It is responsible for the general simulation (e.g. setup, running & recording metrics)"""

import os
from tqdm import tqdm

from src.metrics_keeper import MetricsKeeper
from src.grid import Grid
from src.intersection import Intersection
from src.car_queue import CarQueue
from src.car import Car


def setup_simulation(args):
    """Setup the simulation
    Args:
        args (argparse.Namespace): Arguments parsed from the command line
    Returns:
        Grid: The grid object that contains all intersections and car queues
    """
    grid = Grid(args.grid_size, args.queue_capacity,
                args.unique_auctions, args.auction_modifier_type)
    # Spawn cars in generated grid with given congestion rate
    grid.spawn_cars(args.congestion_rate,
                    args.shared_bid_generator, args.bidders_proportion)

    return grid


def run_epochs(args, grid, metrics_keeper):
    """Run the simulation for the given number of epochs
    Args:
        args (argparse.Namespace): Arguments parsed from the command line
        grid (Grid): The grid object that contains all intersections and car queues
        metrics_keeper (MetricsKeeper): The metrics keeper object that is responsible for recording metrics
    """
    for epoch in tqdm(range(args.num_of_epochs)):
        # Every wage_time epochs, give credit to all cars
        if args.print_grid:
            grid.print_grid(epoch)
        if epoch % args.wage_time == 0:
            give_credit(args)
        # Now that the credit has been given, run the epoch
        run_single_epoch(epoch, grid, metrics_keeper)


def run_single_epoch(epoch, grid, metrics_keeper):
    """Run a single epoch of the simulation
    Args:
        epoch (int): The current epoch number
        grid (Grid): The grid object that contains all intersections and car queues
        metrics_keeper (MetricsKeeper): The metrics keeper object that is responsible for recording metrics
    """
    # First, run auctions & movements
    grid.move_cars()

    # Second, respawn cars that have reached their destination somewhere else, and store their satisfaction scores for evaluation
    satisfaction_scores = grid.respawn_cars(grid.grid_size)
    metrics_keeper.add_satisfaction_scores(epoch, satisfaction_scores)

    # Prepare all entities for the next epoch. This mostly clears epoch-specific variables (e.g. bids submitted)
    grid.ready_for_new_epoch()
    for intersection in Intersection.all_intersections:
        intersection.ready_for_new_epoch()
    for car_queue in CarQueue.all_car_queues:
        car_queue.ready_for_new_epoch()
    for car in Car.all_cars:
        car.ready_for_new_epoch()


def give_credit(args):
    """Give credit to all cars
    Args:
        args (argparse.Namespace): Arguments parsed from the command line
    """
    if Car.all_cars == []:
        print("ERROR: No Cars in Simulation.")
    else:
        for car in Car.all_cars:
            car.set_balance(args.credit_balance)


def reset_all_classes():
    """Reset all classes to their initial state"""
    for intersection in Intersection.all_intersections:
        del intersection
    Intersection.all_intersections = []

    for car_queue in CarQueue.all_car_queues:
        del car_queue
    CarQueue.all_car_queues = []

    for car in Car.all_cars:
        del car
    Car.all_cars = []


def run(args):
    """Main program that runs the simulation
    Args:
        args (argparse.Namespace): Arguments parsed from the command line
    """

    # Create results folder if it doesn't exist
    if not os.path.exists(args.results_folder):
        os.makedirs(args.results_folder)

    metrics_keeper = MetricsKeeper()

    for simulation in range(args.num_of_simulations):
        print(
            f"Running Simulation {simulation + 1} of {args.num_of_simulations}")
        reset_all_classes()
        # Setup the grid on which the simulation will run
        grid = setup_simulation(args)
        # Run the epochs on the grid
        run_epochs(args, grid, metrics_keeper)
        metrics_keeper.prep_for_new_simulation()

    # Produce Results
    metrics_keeper.produce_results(args.results_folder)
