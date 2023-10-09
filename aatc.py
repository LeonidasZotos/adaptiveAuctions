"""Runner Program for the Adaptive Auctions for Traffic Coordination Project."""

import argparse
from datetime import datetime
from math import inf

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
        default=1000,
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
        default=3,
        choices=range(2, 10),
        metavar="[2-9]",
        type=int,
        help="Size of the traffic grid. Defaults to 3 (9 intersections). Must be an integer between 1 and 9."
    )
    run_parser.add_argument(
        "--queue_capacity",
        default=10,
        choices=range(1, 101),
        metavar="[0-100]",
        type=int,
        help="Capacity of each car queue. Defaults to 6. Must be an integer between 1 and 100."
    )

    run_parser.add_argument(
        "--congestion_rate",
        default=0.07,
        type=float_range(0.01, 1),
        help="Rate of congestion (Percentage of occupied spots, 0.01-1)."
    )

    run_parser.add_argument(
        '--with_hotspots',
        action=argparse.BooleanOptionalAction,
        help="""If enabled, there are periodic hotspots to increase congestion in different places.""")

    run_parser.add_argument(
        "--wage_time",
        default=10,
        choices=range(1, 101),
        metavar="[1-100]",
        type=int,
        help="Number of epochs between wage distributions. Defaults to 1. Must be an integer between 1 and 100."
    )
    run_parser.add_argument(
        "--credit_balance",
        default=5,
        type=float_range(1, inf),
        help="Initial & Renewal credit balance for each car. Defaults to infinity. Must be a float between 1 and infinity."
    )

    run_parser.add_argument(
        '--shared_auction_parameters',
        action=argparse.BooleanOptionalAction,
        help="""All auctions will share parameters, instead of each auction having its own parameters.
            As a consequent, the learning algorithm will adjust based on all auctions.""")

    run_parser.add_argument(
        "--adaptive_auction_action_selection",
        default="boltzmann",
        choices=["boltzmann", "e_greedy_decay", "e_greedy_exp_decay",
                 "ucb1", "reverse_sigmoid_decay", "random"],
        type=str,
        help="Type of auction modifier action selection. Defaults to 'e_greedy_decay'. Must be one of 'boltzmann',\
        'e_greedy_decay', 'e_greedy_exp_decay', 'ucb1', 'reverse_sigmoid_decay' or 'random'."
    )

    run_parser.add_argument(
        "--bid_calculation_rule",
        default="linear",
        choices=["linear", "multiplicative", "non-linear"],
        type=str,
        help="This is the way the bid is calculated. If linear, the formula is: bid + (inact_rank * delay_boost). If non-linear, the formula is: bid + (inact_rank/(1-delay_boost)).\
        If multiplicative, the formula is: bid * inact_rank * delay_boost. Defaults to linear. Must be one of 'linear', 'multiplicative' or 'non-linear'."
    )

    run_parser.add_argument(
        "--adaptive_auction_update_rule",
        default="simple_bandit",
        choices=["simple_bandit", "svr"],
        type=str,
        help="This is the rule used to update the expected reward for each parameter combination. E.g. simple_bandit uses a simple average reward, while svr does a fit."
    )

    run_parser.add_argument(
        "--auction_episode_length",
        default=10,
        choices=range(1, 1001),
        metavar="[1-1000]",
        type=int,
        help="Length of an episode for the adaptive auction. I.e. the number of epochs to run the auction with the same parameters. Defaults to 1. Must be an integer between 1 and 1000."
    )
    run_parser.add_argument(
        '--action_selection_hyperparameters',
        nargs="+",  #  At least 1 argument
        type=float,
        help="""Hyperparameters to use for the action selection algorithm. This depends on the algorithm, consult auction_modifier.py for more information. 
                Present to facilitate parameter sweeps. If not present, the default/best found values will be used."""
    )

    run_parser.add_argument(
        "--adaptive_auction_discretization",
        default=25,
        choices=range(1, 1001),
        metavar="[1-1000]",
        type=int,
        help="Number of discrete values to check for each parameter of the adaptive auction. Defaults to 25. Must be an integer between 1 and 100."
    )

    run_parser.add_argument(
        '--only_winning_bid_moves',
        action=argparse.BooleanOptionalAction,
        help="""If enabled, only the car with the wining bid will move (i.e. otherwise, if the 
            winning car can't move, the second highest bidder will move)."""
    )

    run_parser.add_argument(
        "--intersection_reward_type",
        default="mixed_metric_rank",
        choices=["inact_rank", "rank_dist_metric",
                 "mixed_metric_rank", "mixed_rank_dist_metric"],
        type=str,
        help="Type of reward for the intersection. Defaults to 'mixed_metric_rank'. Must be one of 'inact_rank',\
        'mixed_metric_rank', 'mixed_rank_dist_metric' or 'rank_dist_metric'."
    )

    run_parser.add_argument(
        "--inact_rank_weight",
        default=0.5,
        type=float_range(0, 1),
        help="Weight of the inactivity rank used for the intersection auction reward. Defaults to 0.5. Must be a float between 0 and 1."
    )

    run_parser.add_argument(
        "--bid_rank_weight",
        default=0.5,
        type=float_range(0, 1),
        help="Weight of the bid rank used for the intersection auction reward. Defaults to 0.5. Must be a float between 0 and 1."
    )

    run_parser.add_argument(
        '--all_cars_bid',
        action=argparse.BooleanOptionalAction,
        help="""If enabled, all cars of a queue can bid, not only the first one.""")

    run_parser.add_argument(
        '--shared_bid_generator',
        action=argparse.BooleanOptionalAction,
        help="""If enabled, all bidders will share the same bid generator, instead of agents learning to bid separately.""")

    run_parser.add_argument(
        '--bidders_proportion',
        nargs=5,
        default=[1, 1, 0, 0, 0],
        type=int,
        help="""Proportion for the types of bidders to use, should be a list of integers.
            Order: [homogeneous, heterogeneous, random, free-riders & RL]
            Does not have to add up to something. E.g. "2 1 1 0 0 " means 2/4 homogeneous, 1/4 heterogeneous, 1/4 random and 0 free-riders and RL bidders.
            """)

    run_parser.add_argument(
        '--bidders_urgency_distribution',
        default="gaussian",
        choices=["gaussian", "beta"],
        type=str,
        help="Type of distribution from which the bidders's urgency is sampled. Defaults to 'gaussian'. Must be one of 'gaussian' or 'beta'."
    )

    run_parser.add_argument(
        "--results_folder",
        default="results/" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        type=str,
        help="Path to the Results Folder. Defaults to 'results'."
    )

    run_parser.add_argument(
        '--print_grid',
        action=argparse.BooleanOptionalAction,
        help="""If present, the grid will be printed after each epoch.""")
    
    run_parser.add_argument(
        '--sweep_mode',
        action=argparse.BooleanOptionalAction,
        help="""If present, no plots are generated to save time.""")

    # Clean command
    clean_parser = subparsers.add_parser(
        'clean', help='Clean files from previous runs.')

    # Run the appropriate sub-program with arguments
    args = parser.parse_args()
    if args.command == 'run':
        run_simulations.run(args)
    if args.command == 'clean':
        clean.run(args)
