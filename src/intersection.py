from src.car_queue import CarQueue
import random


class Intersection:
    all_intersections = []

    def __init__(self, id, queue_capacity):
        Intersection.all_intersections.append(self)
        self.id = id
        self.carQueues = [CarQueue(queue_capacity, str(id + 'N')),
                          CarQueue(queue_capacity, str(id + 'E')),
                          CarQueue(queue_capacity, str(id + 'S')),
                          CarQueue(queue_capacity, str(id + 'W'))]

    def __str__(self):
        return f'Intersection(id={self.id})'

    def is_empty(self):
        for queue in self.carQueues:
            if not queue.is_empty():
                return False
        return True

    def hold_auction(self):
        # returns the movement from car queue x to car queue y that needs to be executed
        # TODO: implement auction mechanism
        # For now, we pick a random queue to be the winner

        collected_bids = {}
        for queue in self.carQueues:
            if not queue.is_empty():  # Only collect bids from non-empty queues
                collected_bids[queue.id] = queue.collect_bids()
        # Currently we don't use it, but the bids need to be set for later

        winning_queue = random.choice(
            [queue for queue in self.carQueues if not queue.is_empty()])  # Must have cars in queue
        destination = winning_queue.get_destination()
        total_fee = 2  # Placeholder for now
        winning_queue.set_auction_fee(total_fee)
        # We return the originating car queue and the destination car queue. We don't need to know the car ID,
        # as we can retrieve it later, if the move is possible.
        return winning_queue.id, destination

    def ready_for_new_epoch(self):
        # Nothing to clear.
        pass
