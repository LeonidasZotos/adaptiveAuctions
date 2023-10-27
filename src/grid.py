"""This module contains the Grid class. The Grid class is responsible for creating the grid,
(re)spawning cars etc. The Grid contains all intersections, car queues and cars."""

import random
from prettytable import PrettyTable

import src.utils as utils
from src.intersection import Intersection
from src.car import Car
from src.auction_modifier import AuctionModifier
from src.bid_generator import BidGenerator


class Grid:
    """
    The Grid class is responsible for creating the grid, (re)spawning cars etc.
    Attributes:
        args (argparse.Namespace): Arguments parsed from the command line
        all_intersections (list): A list of lists of intersections. The first list represents the rows, the second list represents the columns.
        all_car_queues (list): A list of all car queues in the grid
        all_cars (list): A list of all cars in the grid
        epoch_movements (list): A list of movements that need to be executed in this epoch
    Functions:
        get_all_intersections_and_car_queues: Returns a tuple of all intersections and car queues in the grid
        print_grid(epoch): Prints the grid to the console. The epoch is needed to print the epoch number
        print_cars: Prints all cars to the console
        move_cars: Moves all cars in the grid based on the epoch_movements
        calculate_movements: Calculates the movements that need to be executed in this epoch
        filter_and_execute_movements: Removes movements that are not possible (because the destination queue is full) and 
            executes the movements that are possible, max 1 per intersection
        execute_movement(origin_queue_id, destination_queue_id): Executes a movement (i.e. moves a car from one queue to another)
        spawn_cars(): Spawns cars in the grid with the given congestion rate
        respawn_cars(epoch): Respawns cars that have reached their destination somewhere else. Returns a list of scores, that 
            represent how well the trip went (based on time spent & urgency). Metric used for evaluation.
        ready_for_new_epoch: Clear the class variables that are epoch-specific (e.g. epoch_movements)
    """

    ### General Functions ###
    def __init__(self, args):
        """ Initialize the Grid object
        Args:
            args (argparse.Namespace): Arguments parsed from the command line
        """
        self.args = args

        self.all_intersections = []
        self.all_car_queues = []
        self.all_cars = []
        self.broke_history = []

        intersection_auction_modifier = AuctionModifier(
            self.args, 'same')

        # Create the grid of intersections
        for i in range(self.args.grid_size):
            for j in range(self.args.grid_size):
                # The ID is the x and y coordinates of the intersection
                intersection_id = str(j) + str(i)
                # Each intersection has its own unique auction modifier
                if not self.args.shared_auction_parameters:
                    intersection_auction_modifier = AuctionModifier(
                        self.args, intersection_id)
                intersection = Intersection(
                    self.args, intersection_id, intersection_auction_modifier)
                self.all_car_queues.extend(intersection.get_car_queues())

                self.all_intersections.append(intersection)

        # Keep track of the movements that need to be executed in this epoch
        self.epoch_movements = []

    def __str__(self):
        return f'Grid of size: {self.args.grid_size}, with car queue capacity: {self.args.queue_capacity}'

    ### Getters ###
    def get_all_intersections_and_car_queues(self):
        """Returns a tuple of all intersections and car queues in the grid
        Returns:
            tuple: (list of intersections, list of car queues)
        """
        return self.all_intersections, self.all_car_queues

    def get_percentage_of_broke_agents(self):
        """Returns the percentage of agents that have a balance of 0"""
        type_to_check = "RL"  # "both" or the type of bidder, e.g. "homogeneous"
        percentage = 0
        if type_to_check == "both":
            percentage = len(
                [car for car in self.all_cars if not car.has_balance()]) / len(self.all_cars)
        else:
            # Bidder population must be half/half for this to work properly!!
            percentage = len([car for car in self.all_cars if (not car.has_balance(
            ) and car.bidding_type == type_to_check)]) / (len(self.all_cars)/2)

        return percentage

    def get_broke_history(self):
        """Returns the history of the percentage of agents that have a balance of 0"""
        return self.broke_history

    ### Printing functions ###
    def print_grid(self, epoch):
        """Prints grid in a table format
        Args:
            epoch (int): The current epoch
        """
        print("Grid in epoch: ", epoch)
        grid_table = PrettyTable()
        grid_table.field_names = range(0, self.args.grid_size)
        grid_table.header = False
        grid_table.hrules = True

        for i in range(self.args.grid_size):
            row = []
            for j in range(self.args.grid_size):
                row.append(
                    self.all_intersections[i].get_intersection_description())
            grid_table.add_row(row)
        print(grid_table)

    def print_cars(self):
        """Prints all cars to the console"""
        print("======Start of Cars======")
        for car in self.all_cars:
            print(car)
        print("=======End of Cars=======")

    ### Car spawning functions ###
    def spawn_cars(self):
        """Spawns cars in the grid with the given congestion rate. This should only be exectuted at the start of the simulation
        Returns:
            cars (list): A list of all cars that have been spawned. This is used by the simulator to keep track of all cars.
        """
        # Total spots: Number of Intersections * Number of Queues per intersection (4) * Capacity per queue
        total_spots = self.args.grid_size * \
            self.args.grid_size * 4 * self.args.queue_capacity
        number_of_spawns = int(total_spots * self.args.congestion_rate)
        # As long as spots need to be filled in, spawn cars
        while number_of_spawns > 0:
            # Pick a random intersection and queue
            # We need two randoms as we have a 2D array
            random_intersection = random.choice(self.all_intersections)
            random_queue = random.choice(random_intersection.get_car_queues())
            # If the queue has capacity, spawn a car
            if random_queue.has_capacity():
                bidding_type = random.choices(
                    ['homogeneous', 'heterogeneous', 'random', 'free-rider', 'RL'], weights=self.args.bidders_proportion)[0]
                # If shared_bid_generator is False, create a new BidGenerator object for each car
                # if not self.args.shared_bid_generator:
                bid_generator = BidGenerator(self.args, bidding_type)
                # Create a new car, number_of_spawns is actually the ID.
                car = Car(self.args, number_of_spawns,
                          random_queue, bidding_type, bid_generator)
                # Add the car to the queue
                random_queue.add_car(car)
                # Add the car to the list of all cars
                self.all_cars.append(car)
                # Decrease the number of spawns left
                number_of_spawns -= 1
        return self.all_cars

    def respawn_cars(self, epoch):
        """Respawns cars that have reached their destination somewhere else, with new characteristics (e.g. destination, urgency)
        Args:
            grid_size (int): The size of the grid. This is needed to know which intersections are valid places to spawn cars
        Returns:
            satisfaction_scores (list): A list of small car copies and scores in the form of tuples, that represent
                how well the trip went (based on time spent & urgency). Metric used for evaluation.
        """
        satisfaction_scores = []
        for car in self.all_cars:
            if car.is_at_destination():
                # The time in traffic_network must increase now, as the car has reached its destination and will not go through the 'ready for new epoch' function
                car.time_in_traffic_network += 1
                # If the car is at its destination, remove it from the queue and spawn it somewhere else
                utils.get_car_queue(self.all_car_queues,
                                    car.car_queue_id).remove_car(car)
                # Pick a random queue that has capacity
                random_queue = random.choice(
                    [queue for queue in self.all_car_queues if queue.has_capacity()])
                # Append copy of car and satisfaction score to list
                satisfaction_scores.append(car.calc_satisfaction_score())
                last_satisfaction_score = satisfaction_scores[-1]
                # Reset the car (new destination, new queue, new balance)
                car.reset_car(random_queue.id, epoch, last_satisfaction_score)

                random_queue.add_car(car)
        return satisfaction_scores

    ### Movement functions ###
    def move_cars(self):
        """ Moves all cars in the grid based on the epoch_movements"""
        # First, calculate all movements that need to be made
        self.calculate_movements()
        # Then, filter out movements that are not possible (e.g. because the destination queue is full)
        self.filter_and_execute_movements()

    def calculate_movements(self):
        """ Calculates the movements that need to be executed in this epoch, based on the auction results per intersection"""
        # Request the winning movement from each intersection.
        # Each movement is the originating car queue id and the destination car queue id.
        # Here we have lists of up to 4 tuples, where each tuple represents a movement.
        # In case the top movement can't be made, the next movement is used for that intersection.
        for intersection in self.all_intersections:
            # Only hold an auction if there are cars in the intersection
            if not intersection.is_empty():
                origins, destinations = intersection.hold_auction()
                self.epoch_movements.append((origins, destinations))

    def filter_and_execute_movements(self):
        """Removes movements that are not possible (because the destination queue is full). 
        Executes the movements that are possible, max 1 per intersection.
        The movements are done in a random order, so that no intersection consistently gets priority.
        """
        # First, randomise the order of the movements, so that no intersection consistently gets priority
        random.shuffle(self.epoch_movements)

        # Go through the intersections, and execute up to one movement per intersection
        for intersection_movement in self.epoch_movements:

            intersection = utils.get_intersection_from_car_queue_id(
                self.all_intersections, intersection_movement[0][0])

            for origin_queue_id, destination_queue_id in zip(intersection_movement[0], intersection_movement[1]):
                # If the destination queue is full, remove the movement
                if not utils.get_car_queue(self.all_car_queues, destination_queue_id).has_capacity():
                    # Since the movement is not possible, remove the top fee.
                    intersection.remove_top_fee()
                else:
                    # The movement is possible, so we need to execute it before moving on to the next intersection
                    self.execute_movement(
                        origin_queue_id, destination_queue_id)
                    break

    def execute_movement(self, origin_queue_id, destination_queue_id):
        """Executes a movement (i.e. a car moves from one queue to another)
        Args:
            origin_queue_id (str): The ID of the queue the car is moving from
            destination_queue_id (str): The ID of the queue the car is moving to
        """
        car_queue = utils.get_car_queue(self.all_car_queues, origin_queue_id)
        parent_intersection = car_queue.get_parent_intersection()
        car_queue.win_auction(parent_intersection.get_auction_fee())
        # 1 to signal that there was a car that went through the intersection
        parent_intersection.throughput_history.append(1)

        # Then, all cars must be moved.
        origin_queue = utils.get_car_queue(
            self.all_car_queues, origin_queue_id)
        destination_queue = utils.get_car_queue(
            self.all_car_queues, destination_queue_id)

        car_to_move = origin_queue.remove_first_car()
        car_to_move.increase_distance_travelled_in_trip()
        destination_queue.add_car(car_to_move)
        # Let the car know of its new queue
        car_to_move.set_car_queue_id(destination_queue_id)

    ### Epoch functions ###
    def ready_for_new_epoch(self):
        """ Clears the class variables that are epoch-specific (e.g. epoch_movements)"""
        self.epoch_movements = []
        self.broke_history.append(self.get_percentage_of_broke_agents())
