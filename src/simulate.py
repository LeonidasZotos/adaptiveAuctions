import random
import os
from src.grid import Grid
from src.car import Car
from src.car_queue import CarQueue


def setup_simulation(args):
    grid = Grid(args.grid_size, args.queue_capacity)
    spawn_cars(args)
    give_credit(args)

    for car_queue in CarQueue.all_car_queues:
        print(car_queue)
        
    return


def spawn_cars(args):
    # This should only be exectuted at the start of the simulation
    # Number of Intersections * Number of Queues per intersection (4) * Capacity per queue
    total_spots = args.grid_size * args.grid_size * 4 * args.queue_capacity
    number_of_spawns  = int(total_spots * args.congestion_rate)

    # As long as there are still spots to spawn cars, spawn cars
    while number_of_spawns > 0:
        # Randomly pick a car queue
        queue = CarQueue.all_car_queues[random.randint(0, len(CarQueue.all_car_queues) - 1)]
        # If the queue has capacity, spawn a car
        if queue.has_capacity():
            number_of_spawns -= 1
            print(number_of_spawns)
            queue.add_car(Car(id = number_of_spawns)) # number_of_spawns can be used as a unique ID
            
    return


def give_credit(args):
    # This should only be exectuted once every x iterations
    if Car.all_cars == []:
        print("No Cars in Simulation.")
        return

    for car in Car.all_cars:
        car.set_balance(args.credit_balance)
    print(len(Car.all_cars), " cars have been given credit.")
    return


def run(args):

    if not os.path.exists(args.results_folder):
        os.makedirs(args.results_folder)

    setup_simulation(args)

    print("Simulation Completed")

    return
