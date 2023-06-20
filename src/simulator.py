"""This module contains the Simulator class, which is responsible for running a single simulation."""

from src.metrics import SimulationMetrics
from src.grid import Grid


class Simulator:
    """
    The Simulator class is responsible for running a single simulation, and recording its metrics.
    Attributes:
        id (int): The ID of the simulation
        all_intersections (list): A list of all intersections in the simulation
        all_car_queues (list): A list of all car queues in the simulation
        all_cars (list): A list of all cars in the simulation
        grid (Grid): The grid object that contains all intersections and car queues
        metrics_keeper (SimulationKeeper): The metrics keeper object that is responsible for
            recording metrics just for this simulation
    Functions:
        run_epochs(args): Runs the simulation for the given number of epochs
        run_single_epoch(epoch): Runs a single epoch of the simulation
        run_simulation(args): Runs the simulation
    """

    def __init__(self, args, simulation_id):
        """ Initialize the Simulator object
        Args:
            args (argparse.Namespace): Arguments parsed from the command line
            simulation_id (int): The ID of the simulation
        """
        self.id = simulation_id
        self.all_intersections = []
        self.all_car_queues = []
        self.all_cars = []

        self.grid = Grid(args.grid_size, args.queue_capacity,
                         args.auction_modifier_type, args.intersection_reward)

        # Spawn cars in generated grid with given congestion rate
        self.all_cars = self.grid.spawn_cars(args.congestion_rate,
                                             args.shared_bid_generator, args.bidders_proportion)
        self.all_intersections, self.all_car_queues = self.grid.get_all_intersections_and_car_queues()
        self.metrics_keeper = SimulationMetrics(args, self.grid)

    def run_epochs(self, args):
        """Run the simulation for the given number of epochs
        Args:
            args (argparse.Namespace): Arguments parsed from the command line
        Raises:
            Exception: If there are no cars in the simulation
        """
        for epoch in range(args.num_of_epochs):
            # Every wage_time epochs, give credit to all cars
            if args.print_grid:
                self.grid.print_grid(epoch)
            if epoch % args.wage_time == 0:
                # Give credit to all cars
                if self.all_cars == []:
                    raise Exception("ERROR: No Cars in Simulation.")
                else:
                    for car in self.all_cars:
                        car.set_balance(args.credit_balance)
            # Now that the credit has been given, run the epoch
            self.run_single_epoch(epoch)
        self.metrics_keeper.retrieve_end_of_simulation_metrics()

    def run_single_epoch(self, epoch):
        """Run a single epoch of the simulation
        Args:
            epoch (int): The current epoch number
        """
        # First, run auctions & movements
        self.grid.move_cars()

        # Second, respawn cars that have reached their destination somewhere else, and store their satisfaction scores for evaluation
        satisfaction_scores = self.grid.respawn_cars(self.grid.grid_size)
        self.metrics_keeper.add_satisfaction_scores(epoch, satisfaction_scores)

        # Prepare all entities for the next epoch. This mostly clears epoch-specific variables (e.g. bids submitted)
        self.grid.ready_for_new_epoch()
        for intersection in self.all_intersections:
            intersection.ready_for_new_epoch(epoch)
        for car_queue in self.all_car_queues:
            car_queue.ready_for_new_epoch()
        for car in self.all_cars:
            car.ready_for_new_epoch()
        self.metrics_keeper.ready_for_new_epoch()

    def run_simulation(self, args):
        """Run the simulation
        Args:
            args (argparse.Namespace): Arguments parsed from the command line
        Returns:
            SimulationMetrics: The metrics keeper object that is responsible for recording metrics
        """
        # Run the epochs on the grid
        self.run_epochs(args)
        # Return the metrics keeper, which contains the results of the simulation
        return self.metrics_keeper
