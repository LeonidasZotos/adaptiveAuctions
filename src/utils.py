"""A module containing utility functions, such as finding objects in lists, and removing outliers from data."""


def get_intersection(intersections, intersection_id):
    """Returns the intersection object given an intersection id
    Args:
        intersections (list): A list of all intersections to check
        intersection_id (str): The ID of the intersection (e.g. 11)
    Returns:
        Intersection: The intersection object with the given ID
    Raises:
        Exception: If the intersection ID is not found
    """
    for intersection in intersections:
        if intersection.id == intersection_id:
            return intersection
    else:
        raise Exception(
            "ERROR: Intersection ID not found, with id: ", intersection_id)


def get_car_queue(car_queues, car_queue_id):
    """Returns the car queue object given a car queue id
    Args:
        car_queue_id (str): The ID of the car queue (e.g. 11N)
    Returns:
        CarQueue: The car queue object with the given ID
    Raises:
        Exception: If the car queue ID is not found
    """
    for queue in car_queues:
        if queue.id == car_queue_id:
            return queue
    else:
        raise Exception("ERROR: Queue ID not found, with id: ", car_queue_id)


def get_intersection_from_car_queue_id(intersections, car_queue_id):
    """Returns the intersection object given a car queue id
    Args:
        intersections (list): A list of all intersections to check
        car_queue_id (str): The ID of the car queue (e.g. 11N)
    Returns:    
        Intersection: The intersection object with the given ID
    """
    # Only keep the first two characters of the car_queue_id
    intersection_id = car_queue_id[:2]
    intersection_from_car_queue = get_intersection(
        intersections, intersection_id)
    return intersection_from_car_queue


def get_car_queue_from_intersection(intersection, car_queue_id):
    """Returns the car queue object given a car queue id. Car queue has to be in this intersection.
    Args:
        intersection (Intersection): The intersection object to check
        car_queue_id (str): The ID of the car queue (e.g. 11N)
    Returns:
        CarQueue: The car queue object with the given ID
    Raises:
        Exception: If the car queue ID is not found
    """
    for queue in intersection.carQueues:
        if queue.id == car_queue_id:
            return queue
    else:
        raise Exception("ERROR: Queue ID not found, with id: ", car_queue_id)


def get_last_reward_of_intersection(intersections, intersection_id):
    """Returns the last reward of the intersection
    Args:
        intersections (list): A list of all intersections to check
        intersection_id (str): The ID of the intersection (e.g. 11)
    Returns:
        float: The last reward of the intersection
    """
    return get_intersection(intersections, intersection_id).get_last_reward()


def get_car(cars, car_id):
    """Returns the car object given a car id
    Args:
        car_id (str): The ID of the car (e.g. 5)
    Returns:
        Car: The car object with the given ID
    Raises:
        Exception: If the car ID is not found
    """
    for car in cars:
        if car.id == car_id:
            return car
    else:
        raise Exception("ERROR: Car ID not found, with id: ", car_id)


def remove_outliers(data, restriction=0.1):
    """Removes outliers from a list of data, using the z-score
    Args:
        data (list): A list of data
        restriction (float, optional): The restriction of the outlier removal. Lower restriction leads to more heterogeneous filtering. Defaults to 0.1.
    Returns:
        list: A list of data with outliers removed
    """
    if len(data) == 0:
        return data
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


def smooth_data(lst, window_size=20):
    """Smooths a list of data using a moving average
    Args:
        lst (list): A list of data
        window_size (int, optional): The window size of the moving average. Defaults to 20.
        Returns:
        list: A list of data which has been smoothed
    """
    import math
    smoothed_data = []
    for i in range(len(lst)):
        start = max(0, i - window_size // 2)
        end = min(len(lst), i + window_size // 2 + 1)
        # Filter out None or NA values from the sublist
        sublist = [value for value in lst[start:end]
                   if value is not None and not math.isnan(value)]
        if sublist:
            smoothed_data.append(sum(sublist) / len(sublist))
        else:
            # If all values are None or NA, you can choose to set the result to None.
            smoothed_data.append(None)
    return smoothed_data
