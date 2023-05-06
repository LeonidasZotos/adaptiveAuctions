from src.car_queue import CarQueue
import random


class Intersection:
    all_intersections = []

    def __init__(self, id, queue_capacity):
        Intersection.all_intersections.append(self)
        self.id = id
        self.carQueues = []
        # Create dictionary of car queues for the four positions (e.g. N queue is on the North side),
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
        # For now, we pick a random queue to be the winner
        winning_queue = random.choice([queue for queue in self.carQueues.values(
        ) if not queue.is_empty()])  # Must have cars in queue
        _, destination = winning_queue.get_first_car_and_destination()

        print("winning queue id is ", winning_queue.id,
              "with destination: ", destination)
        print("The type of winning queue id is:", type(winning_queue.id),
              "with destination type: ", type(destination))
        # We return the originating car queue and the destination car queue. We don't need to know the car ID,
        # as we can retrieve it later, if the move is possible.
        return winning_queue.id, destination

    def ready_for_new_epoch(self):
        # Nothing to clear.
        pass
