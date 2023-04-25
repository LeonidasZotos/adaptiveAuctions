from car_queue import CarQueue


class Intersection:
    def __init__(self, id, car_queue_capacity):
        self.id = id
        self.carQueues = []
        # Create dictionary of car queues for the 4 positions (e.g. N queue is on the North side, going E/S/W), 
        self.carQueues = {  # N, S, E, W
            'N': CarQueue(car_queue_capacity),
            'E': CarQueue(car_queue_capacity),
            'S': CarQueue(car_queue_capacity),
            'W': CarQueue(car_queue_capacity)}

    def __str__(self):
        return f'Intersection(id={self.id})'
