"""A module containing utility functions, such as finding class objects by ID"""

from src.intersection import Intersection
from src.car_queue import CarQueue
from src.car import Car


def get_intersection(intersection_id):
    """Returns the intersection object given an intersection id
    Args:
        intersection_id (str): The ID of the intersection (e.g. 11)
    Returns:
        Intersection: The intersection object with the given ID
    """
    for intersection in Intersection.all_intersections:
        if intersection.id == intersection_id:
            return intersection
    print("ERROR: Intersection ID not found, with id: ", intersection_id)


def get_car_queue(car_queue_id):
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


def get_intersection_from_car_queue(car_queue_id):
    """Returns the intersection object given a car queue id
        Args:
        car_queue_id (str): The ID of the car queue (e.g. 11N)
    Returns:    
        Intersection: The intersection object with the given ID
    """
    # only keep the first two characters of the car_queue_id
    intersection_id = car_queue_id[:2]
    intersection_from_car_queue = get_intersection(intersection_id)
    return intersection_from_car_queue


def get_last_reward_of_intersection(intersection_id):
    """Returns the last reward of the intersection
    Args:
        intersection_id (str): The ID of the intersection (e.g. 11)
    Returns:
        float: The last reward of the intersection
    """
    intersection = get_intersection(intersection_id)
    return intersection.get_last_reward()


def get_car(car_id):
    """Returns the car object given a car id
    Args:
        car_id (str): The ID of the car (e.g. 5)
    Returns:
        Car: The car object with the given ID
    """
    for car in Car.all_cars:
        if car.id == car_id:
            return car
    print("ERROR: Car ID not found, with id: ", car_id)
