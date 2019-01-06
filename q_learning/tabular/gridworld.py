# Gridworld example class and helper functions
import matplotlib.pyplot as plt
import numpy as np

class gridworld:
    
    def __init__(self):
        self.dim = [3, 5]
        self.pos_goal = [2, 4]
        self.rew_goal = 10
        self.pos_trap = [0, 4]
        self.rew_trap = -5
        # Define starting position
        self.start = [0, 0]
        self.s = self.start[:]
        self.complete = False
            
        # Step count
        self.n = 0
        self.action_space = [0, 1, 2, 3]
        self.action_dict = {'Up': 0,
                           'Left': 1,
                           'Down': 2,
                           'Right': 3}
        self.action_prob = [0.25, 0.25, 0.25, 0.25]
    
    # Show empty selfironment
    def show_grid(self):
        # print rows
        for i in range(self.dim[0]):
            print("-" * (self.dim[1] * 5 + 1))
            row = []
            for j in range(self.dim[1]):
                if i == self.pos_goal[0] and j == self.pos_goal[1]:
                    row.append("| G ")
                elif i == self.pos_trap[0] and j == self.pos_trap[1]:
                    row.append("| T ")
                elif i == self.start[0] and j == self.start[1]:
                    row.append("| S ")
                else:
                    row.append("|   ")
            row.append("|  ")
            print(' '.join(row))
        print("-" * (self.dim[1] * 5 + 1))
        
    # Show state
    def show_state(self):
        # print rows
        for i in range(self.dim[0]):
            print("-" * (self.dim[1] * 5 + 1))
            row = []
            for j in range(self.dim[1]):
                if i == self.s[0] and j == self.s[1]:
                    row.append("| X ")
                elif i == self.pos_goal[0] and j == self.pos_goal[1]:
                    row.append("| G ")
                elif i == self.pos_trap[0] and j == self.pos_trap[1]:
                    row.append("| T ")
                else:
                    row.append("|   ")
            row.append("|  ")
            print(' '.join(row))
        print("-" * (self.dim[1] * 5 + 1))
        
    # Give the agent an action
    def step(self, a):
        if a not in self.action_space:
            return "Error: Invalid action submission"
        # Check for special terminal states
        if self.s == self.pos_goal:
            self.complete = True
            reward = self.rew_goal
        elif self.s == self.pos_trap:
            self.complete = True
            reward = self.rew_trap
        else:
            # Move up
            if a == 0 and self.s[0] > 0:
                self.s[0] -= 1
            # Move left
            elif a == 1 and self.s[1] > 0:
                self.s[1] -= 1
            # Move down
            elif a == 2 and self.s[0] < self.dim[0] - 1:
                self.s[0] += 1
            # Move right
            elif a == 3 and self.s[1] < self.dim[1] - 1:
                self.s[1] += 1
            reward = -1
        self.n += 1
        return self.s, reward, self.complete
            
    def reset(self):
        self.s = self.start[:]
        self.complete = False
        self.n = 0
        return self.s

    # Plots policy from q-table
    def plot_policy(self, q_table, figsize=(12,8), title='Learned Policy'):
        x = np.linspace(0, self.dim[1] - 1, self.dim[1]) + 0.5
        y = np.linspace(self.dim[0] - 1, 0, self.dim[0]) + 0.5
        X, Y = np.meshgrid(x, y)
        zeros = np.zeros(self.dim)

        fig = plt.figure(figsize=figsize)
        ax = plt.axes()
        # Get max values
        q_max = q_table.max(axis=2)
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                q_star = np.zeros(self.dim)
                q_max_s = q_max[i, j]
                max_vals = np.where(q_max_s==q_table[i,j])[0]
                for action in max_vals:
                    q_star[i,j] = 0.4
                    # Plot results
                    if action == 0:
                        # Move up
                        plt.quiver(X, Y, zeros, q_star, scale=1, units='xy')
                    elif action == 1:
                        # Move left
                        plt.quiver(X, Y, -q_star, zeros, scale=1, units='xy')
                    elif action == 2:
                        # Move down
                        plt.quiver(X, Y, zeros, -q_star, scale=1, units='xy')
                    elif action == 3:
                        # Move right
                        plt.quiver(X, Y, q_star, zeros, scale=1, units='xy')
                        
        plt.xlim([0, self.dim[1]])
        plt.ylim([0, self.dim[0]])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.title(title)
        plt.grid()
        plt.show()