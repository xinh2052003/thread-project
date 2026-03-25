import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from agent import Robot

class GridVisualizer:
    
    def __init__(self, environment, robot):
        self.environment = environment
        self.robot = robot
        self.grid = environment.grid.copy()
        self.grid_size = environment.grid_size
        self.way = []

        self.current_position = (0,0)
        while self.environment.grid[self.current_position] != 0:
            self.current_position = (np.random.randint(self.environment.grid_size), np.random.randint(self.environment.grid_size))
        self.positions = [self.current_position]

    def setStart(self, x, y):
        self.current_position = (x,y)

    def draw_grid(self):
        cmap = mcolors.ListedColormap(['black', 'red', 'white', 'green'])
        bounds = [-1005, -105, 0, 20, 20000]  # Mapped values: bombs, walls, empty, goal
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        plt.figure(figsize=(6, 6))
        plt.imshow(self.grid, cmap=cmap, norm=norm, interpolation='none', extent=[0, self.grid.shape[1], self.grid.shape[0], 0])
        plt.text(self.current_position[0]+0.5, self.current_position[1]+0.5, 'R', color='blue', fontsize=12, ha='center', va='center')

        plt.xticks(np.arange(self.grid.shape[1]))
        plt.yticks(np.arange(self.grid.shape[0]))
        plt.grid(which='both', color='black', linestyle='-', linewidth=0.5)

        plt.show()

    def update(self, new_position):
        self.current_position = new_position
        self.positions.append(new_position)

    
    def get_features(self, position):
        """Lấy các đặc trưng từ vị trí hiện tại"""
        x, y = position  # Tọa độ hiện tại
        grid = self.environment.grid
        grid_size = self.environment.grid_size
        
        # Kiểm tra tường
        wall_top   = True if x > 0 and grid[x-1, y] == -1000 else False
        wall_bot   = True if x < grid_size-1 and grid[x+1, y] == -1000 else False
        wall_left  = True if y > 0 and grid[x, y-1] == -1000 else False
        wall_right = True if y < grid_size-1 and grid[x, y+1] == -1000 else False

        # Kiểm tra bom
        bomb_top   = True if x > 0 and grid[x-1, y] == -10 else False
        bomb_bot   = True if x < grid_size-1 and grid[x+1, y] == -10 else False
        bomb_left  = True if y > 0 and grid[x, y-1] == -10 else False
        bomb_right = True if y < grid_size-1 and grid[x, y+1] == -10 else False

        # Kiểm tra xem di chuyển có gần mục tiêu hơn không
        near_top   = True if x > self.environment.goal_position[0] else False
        near_bot   = True if x < self.environment.goal_position[0] else False
        near_left  = True if y > self.environment.goal_position[1] else False
        near_right = True if y <  self.environment.goal_position[1]else False

        # Lấy hành động kế tiếp
        next_action = self.robot.next_step(position)

        # Trả về từ điển đặc trưng, bao gồm vị trí hiện tại
        return {
            'x': x,  # Tọa độ x hiện tại
            'y': y,  # Tọa độ y hiện tại
            'x_g': self.environment.goal_position[0],
            'y_g':  self.environment.goal_position[1],
            'wall_top': wall_top, 'wall_bot': wall_bot, 'wall_left': wall_left, 'wall_right': wall_right,
            'bomb_top': bomb_top, 'bomb_bot': bomb_bot, 'bomb_left': bomb_left, 'bomb_right': bomb_right,
            'near_top': near_top, 'near_bot': near_bot, 'near_left': near_left, 'near_right': near_right,
            'action': next_action
        }


    def animate(self, frame):
        plt.clf()

        # Show only the current position
        pos = self.positions[frame]
        plt.text(pos[1] + 0.5, pos[0] + 0.5, 'R', color='blue', fontsize=12, ha='center', va='center')
        if pos != self.environment.goal_position:
            if self.grid[pos] == -10:
                self.grid[pos] = 2
            else:
                self.grid[pos] = 1

        light_green = (0.5, 1, 0.8)
        light_red = (1.0, 0.7, 0.7)
        cmap = mcolors.ListedColormap(['black', 'red', 'white', light_green, light_red, 'green'])
        bounds = [-1005, -105, 0, 0.99, 1.99, 20, 20000]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        plt.imshow(self.grid, cmap=cmap, norm=norm, interpolation='none', 
                   extent=[0, self.grid_size, self.grid_size, 0])

        plt.xlim(0, self.grid_size)
        plt.ylim(self.grid_size, 0)
        plt.xticks(np.arange(self.grid_size))
        plt.yticks(np.arange(self.grid_size))
        plt.grid(which='both', color='black', linestyle='-', linewidth=0.5)

    def run_visualization(self):
        try:
            for _ in range(70):
                if self.current_position == self.environment.goal_position:
                    break
                action = self.robot.next_step(self.current_position)

                
                position = self.current_position
                features = self.get_features(position)
                
                self.way.append(features)
                new_position, _= self.environment.step(self.current_position, action)
                self.update(new_position)
            
            fig = plt.figure()
            anim = FuncAnimation(fig, self.animate, frames=len(self.positions), interval=200, repeat=False)
            plt.show()

        except KeyboardInterrupt:
            print("Animation interrupted. Exiting...")  # Handle clean exit if CTRL+C is used
        finally:
            plt.close(fig)