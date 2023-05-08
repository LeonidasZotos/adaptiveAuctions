import os
from src.grid import Grid
from src.intersection import Intersection
from src.car_queue import CarQueue
from src.car import Car


def setup_simulation(args):
    grid = Grid(args.grid_size, args.queue_capacity)
    grid.spawn_cars(args.congestion_rate)
    return grid


def run_epochs(args, grid):
    for i in range(args.num_of_epochs):
        print("Epoch:", i)
        if i % args.wage_time == 0:
            print("Giving credit in epoch:", i)
            give_credit(args)
        else:
            print("Not giving credit in epoch:", i)
        run_single_epoch(grid)


def run_single_epoch(grid):
    # First, run auctions & movements
    grid.move_cars()

    # Second, respawn cars that have reached their destination somewhere else
    grid.respawn_cars(grid.grid_size)

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

    grid = setup_simulation(args)

    run_epochs(args, grid)

    print("Simulation Completed")
