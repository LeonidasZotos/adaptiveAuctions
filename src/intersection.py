from src.car_queue import CarQueue
import random

class Intersection:
    def __init__(self, id, queue_capacity):
        self.id = id
        self.carQueues = []
        # Create dictionary of car queues for the 4 positions (e.g. N queue is on the North side),
        self.carQueues = {  # N, S, E, W
            'N': CarQueue(queue_capacity, str(id + 'N')),
            'E': CarQueue(queue_capacity, str(id + 'E')),
            'S': CarQueue(queue_capacity, str(id + 'S')),
            'W': CarQueue(queue_capacity, str(id + 'W'))
        }

    def __str__(self):
        return f'Intersection(id={self.id})'

    def hold_auction(self):
        # returns the movement from car queue x to car queue y that needs to be executed
        # TODO: implement auction mechanism
        
        # Pick a random queue that has cars in it
        random_queue = random.choice([queue for queue in self.carQueues.values() if not queue.is_empty()])