"""A module containing utility functions, such as finding class objects by ID. The other files are imported 
within the functions, to avoid circular imports."""
from src.intersection import Intersection
from src.car_queue import CarQueue
from src.car import Car

def get_all_intersections():
    """Returns a list of all intersection objects
    Returns:
        list: A list of all intersection objects
    """
    return Intersection.all_intersections

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
    # Only keep the first two characters of the car_queue_id
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


def remove_outliers(data, restriction=0.5):
    """Removes outliers from a list of data, using the z-score
    Args:
        data_list (list): A list of data
        restriction (int, optional): The restriction of the outlier removal. Lower restriction leads to more aggressive filtering. Defaults to 0.5.
    Returns:
        list: A list of data with outliers removed
    """
    mean = sum(data) / len(data)
    standard_deviation = (sum([(i - mean)**2 for i in data]) / len(data))**0.5

    if standard_deviation == 0:
        return data
    z_scores = [(i - mean) / standard_deviation for i in data]

    filtered_data = [i for i, z_score in zip(
        data, z_scores) if abs(z_score) < restriction]
    if len(filtered_data) == 0:
        return data
    return filtered_data
