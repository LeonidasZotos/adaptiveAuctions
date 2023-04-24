
class Car:
    def __init__(self, id):
        self.id = id


    def __str__(self):
        return f'Car(id={self.id})'