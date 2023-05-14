""""This file contains the main simulation loop. It is responsible for the general simulation (e.g. setup, running & recording metrics)"""

import os
from tqdm import tqdm

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

    grid = Grid(args.grid_size, args.queue_capacity)
    # Spawn cars in generated grid with given congestion rate
    grid.spawn_cars(args.congestion_rate)

    return grid


def run_epochs(args, grid):
    """Run the simulation for the given number of epochs
    Args:
        args (argparse.Namespace): Arguments parsed from the command line
        grid (Grid): The grid object that contains all intersections and car queues
    """
    for i in tqdm(range(args.num_of_epochs)):
        # Every wage_time epochs, give credit to all cars
        if i % args.wage_time == 0:
            give_credit(args)
        # Now that the credit has been given, run the epoch
        run_single_epoch(grid)


def run_single_epoch(grid):
    """Run a single epoch of the simulation
    Args:
        grid (Grid): The grid object that contains all intersections and car queues
    """
    # First, run auctions & movements
    grid.move_cars()

    # Second, respawn cars that have reached their destination somewhere else
    grid.respawn_cars(grid.grid_size)

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


def run(args):
    """Main program that runs the simulation
    Args:
        args (argparse.Namespace): Arguments parsed from the command line
    """

    # Create results folder if it doesn't exist
    if not os.path.exists(args.results_folder):
        os.makedirs(args.results_folder)

    # Setup the grid on which the simulation will run
    grid = setup_simulation(args)

    # Run the epochs on the grid
    run_epochs(args, grid)

    print("Simulation Completed")
