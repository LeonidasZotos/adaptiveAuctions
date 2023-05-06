import random
import os
from src.grid import Grid
from src.intersection import Intersection
from src.car_queue import CarQueue
from src.car import Car


def setup_simulation(args):
    grid = Grid(args.grid_size, args.queue_capacity)
    spawn_cars(args)
    give_credit(args)

    return grid


def spawn_cars(args):
    # This should only be exectuted at the start of the simulation
    # Number of Intersections * Number of Queues per intersection (4) * Capacity per queue
    total_spots = args.grid_size * args.grid_size * 4 * args.queue_capacity
    number_of_spawns = int(total_spots * args.congestion_rate)

    # As long as there are still spots to spawn cars, spawn cars
    while number_of_spawns > 0:
        # Randomly pick a car queue
        queue = CarQueue.all_car_queues[random.randint(
            0, len(CarQueue.all_car_queues) - 1)]
        # If the queue has capacity, spawn a car
        if queue.has_capacity():
            number_of_spawns -= 1
            # number_of_spawns can be used as a unique ID
            queue.add_car(
                Car(id=number_of_spawns, car_queue_id=queue.id, grid_size=args.grid_size))
            print("This car queue has id: ", queue.id)


def run_epochs(args, grid):
    for i in range(args.num_epochs):
        print("Epoch: ", i)
        if i % args.wage_time == 0:
            print("Giving credit in epoch ", i)
            give_credit(args)
        else:
            print("Not giving credit in epoch ", i)
        print("Running epoch ", i)
        run_single_epoch(grid)


def run_single_epoch(grid):
    # First, run auctions & movements
    grid.move_cars()

    # Prepare all entities for the next epoch
    grid.ready_for_new_epoch()
    for intersection in Intersection.all_intersections:
        intersection.ready_for_new_epoch()
    for car_queue in CarQueue.all_car_queues:
        car_queue.ready_for_new_epoch()
    for car in Car.all_cars:
        car.ready_for_new_epoch()


def give_credit(args):
    # This should only be exectuted once every x iterations
    if Car.all_cars == []:
        print("No Cars in Simulation.")
    else:
        for car in Car.all_cars:
            car.set_balance(args.credit_balance)
        print(len(Car.all_cars), " cars have been given credit.")


def run(args):

    if not os.path.exists(args.results_folder):
        os.makedirs(args.results_folder)

    setup_simulation(args)
    
    run_epochs(args, setup_simulation(args))

    print("Simulation Completed")
