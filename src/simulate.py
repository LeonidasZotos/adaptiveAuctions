import os


def setupSimulation(args):

    print("Setup of Simulation Completed")


def run(args):

    if not os.path.exists(args.results_folder):
        os.makedirs(args.results_folder)

    print("Simulation Completed")
