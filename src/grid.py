
class Grid:
    def __init__(self, grid_size):
        self.grid_size = grid_size


    def __str__(self):
        return f'Intersection(id={self.id}) of size: {self.grid_size}'