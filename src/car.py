import random


class Car:
    def __init__(self, id):
        self.id = id
        # Randomly pick a direction, from N, S, E, W
        self.direction = random.choice(['N', 'S', 'E', 'W'])
        # Set an initial balance
        self.balance = 100
        # Rush factor is random between 0 and 1
        self.rush_factor = random.random() 
        self.submitted_bid = 0

    def __str__(self):
        return f'Car(id={self.id}), heading {self.direction}'

    def submit_bid(self):
        # For now, randomly submit a bid.
        self.submitted_bid = random.randint(1, 20)
        # If there is not enough balance, bid entire balance.
        if self.submit_bid > self.balance:
            self.submitted_bid = self.balance
        return random.randint(1, 100)

    def pay_bid(self, price):
        # Price is not necessarily the same as the bid (e.g. 2nd price auction)
        self.balance -= price
        print("Car {} paid {} to exit intersection".format(self.id, price))
