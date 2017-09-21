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
                           'Right': 1,
                           'Left': 2,
                           'Down': 3}
        self.action_prob = [0.25, 0.25, 0.25, 0.25]
    
    # Show empty environment
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
    def action(self, a):
        if a not in self.action_space:
            return "Error: Invalid action submission"
        # Check for special terminal states
        if self.s == self.pos_goal:
            self.complete = True
            reward = self.rew_goal
        elif self.s == self.pos_trap:
            self.complete = True
            reward = self.rew_trap
        # Move up
        elif a == 0 and self.s[0] > 0:
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

# Function for viewing the optimal policy of a gridworld based on the state
# values (V(s))
def opt_policy_v(v, grid):
    # Need to define actions that lead to a 
    # higher state value
    
    x = np.linspace(0, grid.dim[1] - 1, grid.dim[1]) + 0.5
    y = np.linspace(grid.dim[0] - 1, 0, grid.dim[0]) + 0.5
    X, Y = np.meshgrid(x, y)
    zeros = np.zeros(grid.dim)

    fig = plt.figure()
    ax = plt.axes()

    # Cycle through each value entry and determine
    # which action leads to a higher value state
    v_star = np.zeros(grid.dim)
    possible_actions = [[-1, 0], # Up = 0
                       [0, 1],  # Right = 1 
                       [1, 0], # Down = 2
                       [0, -1]  # Left = 3
                       ]
    for i in range(grid.dim[0]):
        for j in range(grid.dim[1]):
            v_options = []
            v_star = np.zeros(grid.dim)
            for a_num, a in enumerate(possible_actions):
                coord = [i + a[0], j + a[1]]
                # Ensure action remains within bounds
                if coord[0] >= 0 and coord[0] <= grid.dim[0] - 1:
                    if coord[1] >= 0 and coord[1] <= grid.dim[1] - 1:
                        v_ = v[coord[0], coord[1]]
                        v_options.append([v_, a_num])
                    
            v_options = np.array(v_options)
            max_val = np.max(v_options[:,0])
            pol_rec = v_options[np.where(v_options[:,0]==max_val),1].flatten()
            for direction in pol_rec:
                v_star[i,j] = 0.4
                # Plot results
                if direction == 0:
                    # Vectors point in positive Y-direction
                    plt.quiver(X, Y, zeros, v_star, scale=1, units='xy')
                elif direction == 3:
                    # Vectors point in negative X-direction
                    plt.quiver(X, Y, -v_star, zeros, scale=1, units='xy')
                elif direction == 2:
                    # Vectors point in negative Y-direction
                    plt.quiver(X, Y, zeros, -v_star, scale=1, units='xy')
                elif direction == 1:
                    # Vectors point in positive X-direction
                    plt.quiver(X, Y, v_star, zeros, scale=1, units='xy')
        
    plt.xlim([0, grid.dim[1]])
    plt.ylim([0, grid.dim[0]])
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.grid()
    plt.show()