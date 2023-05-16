""" This module contains the Grid class. The Grid class is responsible for creating the grid,
(re)spawning cars etc. It contains all intersections, car queues and cars."""

import random
from prettytable import PrettyTable

from src.intersection import Intersection
from src.car_queue import CarQueue
from src.car import Car
from src.auction_modifier import AuctionModifier


class Grid:
    """
    The Grid class is responsible for creating the grid, (re)spawning cars etc.
    Attributes:
        grid_size (int): The size of the grid (e.g. 3 means a 3x3 grid)
        queue_capacity (int): The maximum number of cars that can be in a car queue
        intersections (list): A list of lists of intersections. The first list represents the rows, the second list represents the columns.
        epoch_movements (list): A list of movements that need to be executed in this epoch
    Functions:
        print_grid(epoch): Prints the grid to the console. The epoch is needed to print the epoch number
        print_cars(): Prints all cars to the console
        get_car_queue(car_queue_id): Returns the car queue object given a car queue id
        move_cars(): Moves all cars in the grid based on the epoch_movements
        calculate_movements(): Calculates the movements that need to be executed in this epoch
        filter_unfeasible_movements(): Removes movements that are not possible (because the destination queue is full)
        execute_movements(): Executes the movements that are possible
        spawn_cars(congestion_rate): Spawns cars in the grid with the given congestion rate
        respawn_cars(grid_size): Respawns cars that have reached their destination somewhere else
        ready_for_new_epoch(): Clear the class variables that are epoch-specific (e.g. epoch_movements)
    """

    def __init__(self, grid_size, queue_capacity, unique_auctions, auction_modifier_type):
        """ Initialize the Grid object
        Args:
            grid_size (int): The size of the grid (e.g. 3 means a 3x3 grid)
            queue_capacity (int): The maximum number of cars that can be in a car queue
            unique_auctions (bool): Whether each intersection uses its own unique auction modifier, or whether 
                all intersections use the same modifier
            auction_modifier_type (str): The type of the auction modifier(s) (e.g. 'Random', 'Adaptive', 'Static')
        """
        self.grid_size = grid_size
        self.queue_capacity = queue_capacity

        self.intersections = []
        if not unique_auctions:
            auction_modifier = AuctionModifier(auction_modifier_type, "same")
        # Create the grid of intersections
        for i in range(self.grid_size):
            self.intersections.append([])
            for j in range(self.grid_size):
                # The ID is the x and y coordinates of the intersection
                intersection_id = str(j) + str(i)
                if unique_auctions:
                    # Each intersection has its own unique auction modifier
                    intersection_auction_modifier = AuctionModifier(
                        auction_modifier_type, intersection_id)
                    self.intersections[i].append(Intersection(
                        intersection_id, self.queue_capacity, intersection_auction_modifier))
                else:
                    # All intersections use the same auction modifier, created before the loop
                    self.intersections[i].append(Intersection(
                        intersection_id, self.queue_capacity, auction_modifier))

        # Keep track of the movements that need to be executed in this epoch
        self.epoch_movements = []

    def __str__(self):
        return f'Grid of size: {self.grid_size}, with car queue capacity: {self.queue_capacity}'

### Printing functions ###
    def print_grid_simple(self):
        """Prints the grid to the console
        """
        print("======Start of Grid======")
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                print(self.intersections[i][j], end="\n")
                for queue in self.intersections[i][j].carQueues:
                    print(queue, end=" ")
                    print()
                    for car in queue.cars:
                        print(car, end=" ")
                        print()
            print()
        print("=======End of Grid=======")

    def print_grid(self, epoch):
        """Prints grid in a table format
        Args:
            epoch (int): The current epoch
        """
        print("Grid in epoch: ", epoch)
        grid_table = PrettyTable()
        grid_table.field_names = range(0, self.grid_size)
        grid_table.header = False
        grid_table.hrules = True

        for i in range(self.grid_size):
            row = []
            for j in range(self.grid_size):
                row.append(self.intersections[i]
                           [j].get_intersection_description())
            grid_table.add_row(row)
        print(grid_table)

    def print_cars(self):
        """Prints all cars to the console"""
        print("======Start of Cars======")
        for car in Car.all_cars:
            print(car)
        print("=======End of Cars=======")

### Helper functions ###
    def get_car_queue(self, car_queue_id):
        """Returns the car queue object given a car queue id
        Args:
            car_queue_id (str): The ID of the car queue (e.g. 11N)
        Returns:
            CarQueue: The car queue object with the given ID
        """
        for queue in CarQueue.all_car_queues:
            if queue.id == car_queue_id:
                return queue
        print("ERROR: Queue ID not found, with id: ", car_queue_id)

### Movement functions ###
    def move_cars(self):
        """ Moves all cars in the grid based on the epoch_movements
        """
        # First, calculate all movements that need to be made
        self.calculate_movements()
        # Then, filter out movements that are not possible (e.g. because the destination queue is full)
        self.filter_unfeasible_movements()
        # Finally, execute the movements that are possible
        self.execute_movements()

    def calculate_movements(self):
        """ Calculates the movements that need to be executed in this epoch, based on the auction results per intersection
        """
        # Request the winning movement from each intersection.
        # Each movement is the originating car queue id and the destination car queue id.
        for intersection in Intersection.all_intersections:
            # Only hold an auction if there are cars in the intersection
            if not intersection.is_empty():
                origin, destination = intersection.hold_auction()
                self.epoch_movements.append((origin, destination))

    def filter_unfeasible_movements(self):
        """Removes movements that are not possible (because the destination queue is full).
        If there is more demand than capacity for a destination queue, remove random movements until demand is met.
        The capacity is calculated without considering departing cars, as they will be removed from the queue before the new cars arrive.
        """
        # First we need to know the capacity of each destination queue
        queues_and_their_capacities = {}
        for _, destination_queue_id in self.epoch_movements:
            queues_and_their_capacities[destination_queue_id] = self.get_car_queue(
                destination_queue_id).get_num_of_free_spots()

        # Second, we need to know the demand for each destination queue
        queues_and_their_demand = {}
        for _, destination_queue_id in self.epoch_movements:
            if destination_queue_id not in queues_and_their_demand.keys():
                queues_and_their_demand[destination_queue_id] = 1
            elif destination_queue_id in queues_and_their_demand.keys():
                queues_and_their_demand[destination_queue_id] += 1

        # Delete random movements, so that the demand is met (not more movements than capacity of destination queue)
        for queue_id, demand in queues_and_their_demand.items():
            # If there is more demand than capacity, remove random movements until demand is met
            if demand > queues_and_their_capacities[queue_id]:
                # List of all movements that go to this queue
                movements_to_this_queue = [
                    movement for movement in self.epoch_movements if movement[1] == queue_id]
                # Remove random movements destined to this queue until demand is met
                while demand > queues_and_their_capacities[queue_id]:
                    # Pick a random movement to remove
                    movement_to_remove = random.choice(movements_to_this_queue)
                    self.epoch_movements.remove(
                        movement_to_remove)
                    # Update the demand
                    demand -= 1
                    # Update the list of movements to this queue
                    movements_to_this_queue = [
                        movement for movement in self.epoch_movements if movement[1] == queue_id]

    def execute_movements(self):
        """Execute the movements that are possible (cars need to pay the auction fee, and then move to the destination queue)
        """
        # First, all winning car queues must pay the bid and update their inactivity (though win_auction())
        for movement in self.epoch_movements:
            oringin_queue_id, destination_queue_id = movement
            self.get_car_queue(oringin_queue_id).win_auction()

        # Then, all cars must be moved. This is done after to avoid having winning cars that end up in winning queues paying the fee twice.
        # NOTE: This might not be a problem as the bids are stored by the car queues themselves???
        for movement in self.epoch_movements:
            oringin_queue_id, destination_queue_id = movement
            origin_queue = self.get_car_queue(oringin_queue_id)
            destination_queue = self.get_car_queue(destination_queue_id)

            car_to_move = origin_queue.remove_first_car()
            destination_queue.add_car(car_to_move)
            # Let the car know of its new queue
            car_to_move.set_car_queue_id(destination_queue_id)

### Car spawning functions ###
    def spawn_cars(self, congestion_rate):
        """Spawns cars in the grid with the given congestion rate. This should only be exectuted at the start of the simulation
        Args:
            congestion_rate (float): The congestion rate of the grid (e.g. 0.5 means 50% of the spots are occupied)
        """
        # Total spots: Number of Intersections * Number of Queues per intersection (4) * Capacity per queue
        total_spots = self.grid_size * self.grid_size * 4 * self.queue_capacity
        number_of_spawns = int(total_spots * congestion_rate)

        # As long as spots need to be filled in, spawn cars
        while number_of_spawns > 0:
            # Randomly pick a car queue
            queue = CarQueue.all_car_queues[random.randint(
                0, len(CarQueue.all_car_queues) - 1)]
            # If the queue has capacity, spawn a car
            if queue.has_capacity():
                number_of_spawns -= 1
                # number_of_spawns can be used as a unique ID
                queue.add_car(
                    Car(id=number_of_spawns, car_queue_id=queue.id, grid_size=self.grid_size))

    def respawn_cars(self, grid_size):
        """Respawns cars that have reached their destination somewhere else, with new characteristics (e.g. destination, rush_factor)
        Args:
            grid_size (int): The size of the grid. This is needed to know which intersections are valid places to spawn cars
        """
        for car in Car.all_cars:
            if car.is_at_destination():
                # If the car is at its destination, remove it from the queue and spawn it somewhere else
                self.get_car_queue(car.car_queue_id).remove_car(car)
                # Pick a random queue that has capacity
                random_queue = random.choice(
                    [queue for queue in CarQueue.all_car_queues if queue.has_capacity()])
                # Reset the car (new destination, new queue, new balance)
                car.reset_car(random_queue.id, grid_size)
                random_queue.add_car(car)

### Epoch functions ###
    def ready_for_new_epoch(self):
        """ Clears the class variables that are epoch-specific (e.g. epoch_movements)
        """
        self.epoch_movements = []
