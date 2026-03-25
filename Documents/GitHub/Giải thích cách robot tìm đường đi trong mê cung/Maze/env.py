import numpy as np

class GridEnvironment:
    def __init__(self, grid_size = 10, num_bomb = -1, num_wall = -1):
        self.grid_size = grid_size
        self.num_bombs = grid_size if num_bomb == -1 else num_bomb
        self.num_walls = grid_size * 2 if num_wall == -1 else num_wall
        self.reset_environment()

    def show_grid(self):
        for row in self.grid:
            print(" ".join(str(e) for e in row))

    def reset_environment(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        self.goal_position = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
        self.grid[self.goal_position] = 100

        for _ in range(self.num_walls):
            pos = self._get_random_position(exclude=[self.goal_position])
            self.grid[pos] = -1000

        for _ in range(self.num_bombs):
            pos = self._get_random_position(exclude=[self.goal_position])
            self.grid[pos] = -10
    

    

    def _get_random_position(self, exclude=[]):
        pos = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
        while pos in exclude:
            pos = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
        return pos


    def step(self, position, action):
        if action == 'up':
            if position[0] == 0: 
                return position, -10000
            else:
                new_position = (position[0] - 1, position[1])

        elif action == 'down':
            if position[0] == self.grid_size - 1:
                return position, -10000
            else:
                new_position = (position[0] + 1, position[1])

        elif action == 'left':
            if position[1] == 0:
                return position, -10000
            else:
                new_position = (position[0], position[1] - 1)

        elif action == 'right':
            if position[1] == self.grid_size - 1:
                return position, -10000
            else:
                new_position = (position[0], position[1] + 1)

        reward = self.grid[new_position]
        if reward == -1:
            new_position = position

        return new_position, reward
    
    def print_maze(self):
        for row in self.grid:
            # Chuyển đổi mỗi giá trị thành ký tự
            display_row = []
            for cell in row:
                if cell == 100:
                    display_row.append('G')  # Mục tiêu
                elif cell == -1000:
                    display_row.append('W')  # Tường
                elif cell == -10:
                    display_row.append('B')  # Bom
                else:
                    display_row.append('.')  # Ô trống
            print(" ".join(display_row))