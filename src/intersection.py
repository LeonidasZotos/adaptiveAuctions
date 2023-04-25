from src.car_queue import CarQueue


class Intersection:
    def __init__(self, id, queue_capacity):
        self.id = id
        self.carQueues = []
        # Create dictionary of car queues for the 4 positions (e.g. N queue is on the North side, going E/S/W),
        self.carQueues = {  # N, S, E, W
            'N': CarQueue(queue_capacity),
            'E': CarQueue(queue_capacity),
            'S': CarQueue(queue_capacity),
            'W': CarQueue(queue_capacity)
        }

    def __str__(self):
        return f'Intersection(id={self.id})'
