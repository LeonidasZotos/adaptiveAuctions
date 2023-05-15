"""Runner Program for the Adaptive Auctions for Traffic Coordination Project."""

import argparse

from src import simulate, test, clean

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

    # The 3 subparsers are: simulate, test & clean
    subparsers = parser.add_subparsers(
        help='commands', title="commands", dest="command")

    # Simulate command
    simulate_parser = subparsers.add_parser(
        'simulate', help='Run a full Simulation')
    simulate_parser.add_argument(
        "--grid_size",
        default=4,
        choices=range(2, 10),
        metavar="[2-100]",
        type=int,
        help="Size of the traffic grid. Defaults to 3 (9 intersections). Must be an integer between 1 and 10."
    )
    simulate_parser.add_argument(
        "--queue_capacity",
        default=4,
        choices=range(1, 101),
        metavar="[0-100]",
        type=int,
        help="Capacity of each car queue. Defaults to 4. Must be an integer between 1 and 100."
    )

    simulate_parser.add_argument(
        "--congestion_rate",
        default=0.5,
        type=float_range(0.1, 1),
        help="Rate of congestion (Percentage of occupied spots, 0.1-1). Defaults to 0.5. Must be a float between 0.1 and 1."
    )

    simulate_parser.add_argument(
        "--credit_balance",
        default=30,
        type=float_range(1, 1000),
        help="Initial & Renewal credit balance for each car. Defaults to 30. Must be a float between 1 and 1000."
    )

    simulate_parser.add_argument(
        "--num_of_epochs",
        default=100,
        choices=range(1, 5001),
        metavar="[1-5000]",
        type=int,
        help="Number of epochs to run. Defaults to 5. Must be an integer between 1 and 5000."
    )

    simulate_parser.add_argument(
        "--wage_time",
        default=2,
        choices=range(1, 5001),
        metavar="[1-5000]",
        type=int,
        help="Number of epochs between wage distributions. Defaults to 2. Must be an integer between 1 and 5000."
    )

    simulate_parser.add_argument(
        "--results_folder",
        default="results",
        type=str,
        help="Path to the Results Folder. Defaults to 'results'."
    )

    simulate_parser.add_argument(
        "--auction_modifier_type",
        default="random",
        choices=["random", "static", "spsa"],
        type=str,
        help="Type of auction modifier. Defaults to 'random'. Must be one of 'random', 'static' or 'spsa'."
    )

    simulate_parser.add_argument(
        '--unique_auctions', action=argparse.BooleanOptionalAction,
        help="""If present, each intersection will use different auction modifiers, and by extension,
            potentially different parameters. Otherwise, all intersections will use the same auction modifiers.""")

    # Test command
    test_parser = subparsers.add_parser(
        'test', help='Test the program with different parameters')

    # Clean command
    clean_parser = subparsers.add_parser(
        'clean', help='Clean files from previous runs.')

    # Run the appropriate sub-program with arguments
    args = parser.parse_args()
    if args.command == 'simulate':
        simulate.run(args)
    if args.command == 'clean':
        clean.run(args)
    if args.command == 'test':
        test.run(args)
