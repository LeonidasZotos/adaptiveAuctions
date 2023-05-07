from src.intersection import Intersection
from src.car_queue import CarQueue
import random


class Grid:

    def __init__(self, grid_size, queue_capacity):
        self.grid_size = grid_size
        self.queue_capacity = queue_capacity
        self.map = self.create_grid()
        # Keep track of the movements that need to be executed in this epoch
        self.epoch_movements = []

    def __str__(self):
        return f'Grid of size: {self.grid_size}, with car queue capacity: {self.queue_capacity}'

    def create_grid(self):
        grid = []
        for i in range(self.grid_size):
            grid.append([])
            for j in range(self.grid_size):
                # The ID is the x and y coordinates of the intersection
                intersection_id = str(j) + str(i)
                grid[i].append(Intersection(
                    intersection_id, self.queue_capacity))
        return grid

    def print_grid(self):
        print("Printing Grid:")
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                print(self.map[i][j], end=" ")
            print()

    def move_cars(self):
        self.calculate_movements()
        self.filter_unfeasible_movements()
        self.execute_movements()

    def calculate_movements(self):
        # Request the winning movement from each intersection.
        # Each movement is the originating car queue id and the destination car queue id.
        for intersection in Intersection.all_intersections:
            origin, destination = intersection.hold_auction()
            self.epoch_movements.append((origin, destination))

    def filter_unfeasible_movements(self):
        # Remove movements that are not possible (Because the destination queue is full)
        queues_and_their_capacities = {}
        for _, destination_queue_id in self.epoch_movements:
            queues_and_their_capacities[destination_queue_id] = self.get_car_queue(
                destination_queue_id).get_num_of_free_spots()

        queues_and_their_demand = {}
        for _, destination_queue_id in self.epoch_movements:
            if destination_queue_id not in queues_and_their_demand.keys():
                queues_and_their_demand[destination_queue_id] = 1
            elif destination_queue_id in queues_and_their_demand.keys():
                queues_and_their_demand[destination_queue_id] += 1

        # Delete random movements, so that there is one movement per queue
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

    def get_car_queue(self, car_queue_id):
        # Returns the car queue object given a car queue id
        for queue in CarQueue.all_car_queues:
            if queue.id == car_queue_id:
                return queue
        print("ERROR: Queue ID not found, with id: ", car_queue_id)

    def execute_movements(self):
        # Execute the movements that are possible
        # First, all winning car queues must pay the bid and update their inactivity (though win_auction())
        for movement in self.epoch_movements:
            oringin_queue_id, destination_queue_id = movement
            self.get_car_queue(oringin_queue_id).win_auction()

        # Then, all cars must be moved. This has to be done after the payment, because otherwise if a winning car moves to another winning queue, they might pay twice
        for movement in self.epoch_movements:
            oringin_queue_id, destination_queue_id = movement
            origin_queue = self.get_car_queue(oringin_queue_id)
            destination_queue = self.get_car_queue(destination_queue_id)
            car_to_move = origin_queue.remove_first_car()
            destination_queue.add_car(car_to_move)
            # Let the car know of its new queue
            car_to_move.update_car_queue_id(destination_queue_id)

    def ready_for_new_epoch(self):
        self.epoch_movements = []
