import numpy as np
import pandas as pd  # Để xử lý dữ liệu và xuất ra CSV

class Robot:
    def __init__(self, environment, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.environment = environment
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((environment.grid_size, environment.grid_size, 4))
        self.actions = ['up', 'down', 'left', 'right']
        self.dataset = []  # Danh sách lưu dữ liệu đặc trưng

    def choose_action(self, position):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.actions)
        else:
            state_actions = self.q_table[position[0], position[1], :]
            return self.actions[np.argmax(state_actions)]
        
    def next_step(self, position):
        state_actions = self.q_table[position[0], position[1], :]
        return self.actions[np.argmax(state_actions)]

    def learn(self, current_position, action, reward, next_position):
        current_state_action = self.q_table[current_position[0], current_position[1], self.actions.index(action)]
        next_state_action = np.max(self.q_table[next_position[0], next_position[1], :])
        q_target = reward + self.gamma * next_state_action
        self.q_table[current_position[0], current_position[1], self.actions.index(action)] = \
            current_state_action + self.alpha * (q_target - current_state_action)

    
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
        next_action = self.next_step(position)

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

    def train(self, episodes=4000):
        for x in range(episodes):
            position = (np.random.randint(self.environment.grid_size), np.random.randint(self.environment.grid_size))
            while self.environment.grid[position] != 0:
                position = (np.random.randint(self.environment.grid_size), np.random.randint(self.environment.grid_size))

            

            cnt = 0
            while position != self.environment.goal_position:
                # Thu thập đặc trưng tại vị trí hiện tại, bao gồm vị trí và next_action
                
                if x >= 2000: 
                    features = self.get_features(position)
                    self.dataset.append(features)

                action = self.choose_action(position)  # Hành động thực tế trong huấn luyện
                next_position, reward = self.environment.step(position, action)
                self.learn(position, action, reward, next_position)
                position = next_position
                cnt += 1
                if cnt == 100:
                    break

        # Lưu dữ liệu vào file CSV sau khi huấn luyện
        df = pd.DataFrame(self.dataset)
        df.to_csv('dataset.csv', index=False)
        print("Dataset saved to 'dataset.csv'")