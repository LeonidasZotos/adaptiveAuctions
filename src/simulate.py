import os
from src.grid import Grid


def setupSimulation(args):

    print("Setup of Simulation Completed")


def run(args):

    if not os.path.exists(args.results_folder):
        os.makedirs(args.results_folder)

    grid = Grid(args.grid_size, args.queue_capacity)
    grid.print_grid()

    print("Simulation Completed")
