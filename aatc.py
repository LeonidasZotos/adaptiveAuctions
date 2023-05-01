"""Runner Program for the Adaptive Auctions for Traffic Coordination Project. 
"""

import argparse
from src import simulate, clean

# Utility Functions
def float_range(mini, maxi):
    """Return function handle of an argument type function for 
       ArgumentParser checking a float range: mini <= arg <= maxi
       Source: https://stackoverflow.com/a/64259328
         mini - minimum acceptable argument
         maxi - maximum acceptable argument
        """

    # Define the function with default arguments
    def float_range_checker(arg):
        """New Type function for argparse - a float within predefined range."""

        try:
            f = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError("must be a floating point number")
        if f < mini or f > maxi:
            raise argparse.ArgumentTypeError(
                "must be in range [" + str(mini) + " .. " + str(maxi)+"]")
        return f

    # Return function handle to checking function
    return float_range_checker


# Main Function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="Adaptive Auctions for Traffic Coordination",
        description="Welcome to the 'Adaptive Auctions for Traffic Coordination' program."
    )

    # The 2 subparsers are: simulate, test
    subparsers = parser.add_subparsers(
        help='commands', title="commands", dest="command")

    # Simulate command
    simulate_parser = subparsers.add_parser(
        'simulate', help='Run a full Simulation')
    simulate_parser.add_argument(
        "--grid_size",
        default=3,
        options=range(1, 100),
        type=int,
        help="Size of the traffic grid. Defaults to 3 (9 intersections). Must be an integer between 1 and 100."
    )
    simulate_parser.add_argument(
        "--queue_capacity",
        default=4,
        options=range(1, 100),
        type=int,
        help="Capacity of each car queue. Defaults to 4. Must be an integer between 1 and 100."
    )

    simulate_parser.add_argument(
        "--congestion_rate",
        default=0.5,
        options=float_range(0, 1),
        type=float,
        help="Rate of congestion (Percentage of occupied spots, 0-1). Defaults to 0.5. Must be a float between 0 and 1."
    )

    simulate_parser.add_argument(
        "--credit_balance",
        default=30,
        options = float_range(1, 1000),
        type=float,
        help="Initial & Renewal credit balance for each car. Defaults to 30. Must be a float between 1 and 1000."
    )

    simulate_parser.add_argument(
        "--results_folder",
        default="results",
        type=str,
        help="Path to the Results Folder. Defaults to 'results'."
    )

    # Clean command
    clean_parser = subparsers.add_parser(
        'clean', help='Clean files from previous runs.')

    # Run the appropriate sub-program with arguments
    args = parser.parse_args()
    if args.command == 'simulate':
        simulate.run(args)
    if args.command == 'clean':
        clean.run(args)
