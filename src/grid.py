from src.intersection import Intersection


class Grid:
    def __init__(self, grid_size, queue_capacity):
        self.grid_size = grid_size
        self.queue_capacity = queue_capacity
        self.map = self.create_grid()

    def __str__(self):
        return f'Grid of size: {self.grid_size}, with car queue capacity: {self.queue_capacity}'

    def create_grid(self):
        grid = []
        for i in range(self.grid_size):
            grid.append([])
            for j in range(self.grid_size):
                # The id is the x and y coordinates of the intersection
                intersection_id = str(j) + str(i)
                grid[i].append(Intersection(
                    intersection_id, self.queue_capacity))
        return grid

    def print_grid(self):
        print("Printing Grid:")
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                print(self.map[i][j], end=" ")
            print()

        pass
