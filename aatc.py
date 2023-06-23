"""Runner Program for the Adaptive Auctions for Traffic Coordination Project."""

import argparse
from datetime import datetime

from src import run_simulations, clean

# Utility Functions


def float_range(minimum, maximum):
    """Return function handle of an argument type function for
       ArgumentParser checking a float range: minimum <= arg <= maximum
       Source: https://stackoverflow.com/a/64259328
        Args:
            minimum - minimum acceptable argument
            maximum - maximum acceptable argument
        Returns:
            function handle to checking function
        """

    # Define the function with default arguments
    def float_range_checker(arg):
        """New Type function for argparse - a float within predefined range."""

        try:
            f = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError("Must be a floating point number")
        if f < minimum or f > maximum:
            raise argparse.ArgumentTypeError(
                "Must be in range [" + str(minimum) + "-" + str(maximum)+"]")
        return f

    # Return function handle to checking function
    return float_range_checker


# Main Function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="Adaptive Auctions for Traffic Coordination",
        description="Welcome to the 'Adaptive Auctions for Traffic Coordination' program."
    )

    # The 2 subparsers are: run & clean
    subparsers = parser.add_subparsers(
        help='commands', title="commands", dest="command")

    # Simulate command
    run_parser = subparsers.add_parser(
        'run', help='Run a full Simulation')

    run_parser.add_argument(
        "--num_of_epochs",
        default=500,
        choices=range(1, 100001),
        metavar="[1-100000]",
        type=int,
        help="Number of epochs to run. Defaults to 200. Must be an integer between 1 and 5000."
    )

    run_parser.add_argument(
        "--num_of_simulations",
        default=50,
        choices=range(1, 10001),
        metavar="[1-10000]",
        type=int,
        help="Number of simulations to run. Defaults to 1. Must be an integer between 1 and 1000."
    )

    run_parser.add_argument(
        "--grid_size",
        default=5,
        choices=range(2, 10),
        metavar="[2-100]",
        type=int,
        help="Size of the traffic grid. Defaults to 3 (9 intersections). Must be an integer between 1 and 10."
    )
    run_parser.add_argument(
        "--queue_capacity",
        default=5,
        choices=range(1, 101),
        metavar="[0-100]",
        type=int,
        help="Capacity of each car queue. Defaults to 6. Must be an integer between 1 and 100."
    )

    run_parser.add_argument(
        "--congestion_rate",
        default=0.15,
        type=float_range(0.01, 1),
        help="Rate of congestion (Percentage of occupied spots, 0.01-1). Defaults to 0.5. Must be a float between 0.1 and 1."
    )

    run_parser.add_argument(
        "--credit_balance",
        default=100,
        type=float_range(1, 1000),
        help="Initial & Renewal credit balance for each car. Defaults to 50. Must be a float between 1 and 1000."
    )

    run_parser.add_argument(
        "--wage_time",
        default=5,
        choices=range(1, 101),
        metavar="[1-100]",
        type=int,
        help="Number of epochs between wage distributions. Defaults to 20. Must be an integer between 1 and 5000."
    )

    run_parser.add_argument(
        '--shared_auction_parameters',
        action=argparse.BooleanOptionalAction,
        help="""All auctions will share parameters, instead of each auction having its own parameters. 
            As a consequent, SPSA will adjust based on all auctions.""")

    run_parser.add_argument(
        "--auction_modifier_type",
        default="static",
        choices=["random", "static", "spsa"],
        type=str,
        help="Type of auction modifier. Defaults to 'static'. Must be one of 'random', 'static' or 'spsa'."
    )

    run_parser.add_argument(
        "--intersection_reward",
        default="time_and_urgency",
        choices=["time", "time_and_urgency"],
        type=str,
        help="Type of reward for the intersection. Defaults to 'time_and_urgency'. Must be one of 'time' or 'time_and_urgency'."
    )

    run_parser.add_argument(
        '--shared_bid_generator',
        action=argparse.BooleanOptionalAction,
        help="""Each car will learn how to bid individually, instead of sharing experiences.""")

    run_parser.add_argument(
        '--bidders_proportion',
        nargs=4,
        default=[1, 1, 1, 0],
        type=int,
        help="""Proportion for the types of bidders to use, should be a list of integers.
            Order: [static, random, free-riders & RL]
            Does not have to add up to something. E.g. "2 1 1 0" means 2/4 static, 1/4 random, 1/4 free-riders and 0 RL bidders.
            """)

    run_parser.add_argument(
        "--results_folder",
        # include time and date in the folder name
        default="results/" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        type=str,
        help="Path to the Results Folder. Defaults to 'results'."
    )

    run_parser.add_argument(
        '--print_grid',
        action=argparse.BooleanOptionalAction,
        help="""If present, the grid will be printed after each epoch.""")

    # Clean command
    clean_parser = subparsers.add_parser(
        'clean', help='Clean files from previous runs.')

    # Run the appropriate sub-program with arguments
    args = parser.parse_args()
    if args.command == 'run':
        run_simulations.run(args)
    if args.command == 'clean':
        clean.run(args)
