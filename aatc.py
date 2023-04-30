"""Runner Program for the Adaptive Auctions for Traffic Coordination Project. 
"""

import argparse
from src import simulate, clean

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
        type=int,
        help="Size of the traffic grid. Defaults to 3 (9 intersections)."
    )
    simulate_parser.add_argument(
        "--queue_capacity",
        default=4,
        type=int,
        help="Capacity of each car queue. Defaults to 4."
    )

    simulate_parser.add_argument(
        "--congestion_rate",
        default=0.5,
        type=float,
        help="Rate of congestion (Percentage of occupied spots, 0-1). Defaults to 0.5."
    )

    simulate_parser.add_argument(
        "--credit_balance",
        default=30,
        type=float,
        help="Initial & Renewal credit balance for each car. Defaults to 30."
    )

    simulate_parser.add_argument(
        "--results_folder",
        default="results",
        type=str,
        help="Path to the Results Folder. Defaults to 'results'."
    )

    # Test command
    clean_parser = subparsers.add_parser(
        'clean', help='Clean files from previous runs.')

    # Run the appropriate sub-program with arguments
    args = parser.parse_args()
    if args.command == 'simulate':
        simulate.run(args)
    if args.command == 'clean':
        clean.run(args)
