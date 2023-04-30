import random


class Car:
    # A class variable to keep track of all cars.
    all_cars = []

    def __init__(self, id):
        Car.all_cars.append(self)
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
        if self.submitted_bid > self.balance:
            self.submitted_bid = self.balance

        # Return a tuple of the car ID and the bid.
        # Car ID is included so that the intersection can keep track of which car submitted which bid.
        return self.id, self.submitted_bid

    def pay_bid(self, price):
        # Price is not necessarily the same as the bid (e.g. 2nd price auction)
        self.balance -= price
        print("Car {} paid {} to exit intersection".format(self.id, price))
        
    def set_balance(self, balance):
        # Set the balance of the car to the given balance
        self.balance = balance
