import numpy as np 
from sklearn.datasets import make_moons 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt

class network():
    
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.n_input_dim = X.shape[1]
        self.n_output_dim = 1
        self.n_inputs = X.shape[0]
        
    def initialize(self, n_hidden, seed=1):
        self.n_hidden = n_hidden
        np.random.seed(seed)
        c = np.sqrt(3 / (0.5 + self.n_input_dim + self.n_output_dim))
        W1 = np.random.uniform(low=-c, high=c, 
                    size=(self.n_input_dim, self.n_hidden))
        b1 = np.zeros((1, self.n_hidden))
        W2 = np.random.uniform(low=-c, high=c,
                              size=(self.n_hidden, self.n_output_dim))
        b2 = np.zeros((1, self.n_output_dim)) 
        
        self.params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        self.cache = {}
        
    def Elu(self, x, a=2):
        return np.where(x<=0, (a * (np.exp(x) - 1)), x)
    
    def dElu(self, x, a=2):
        return np.where(x<=0, a * np.exp(x), 1)
    
    def sigmoid(self, x):
        return 1. / (1 + np.exp(-x))
    
    def forward_prop(self, X=None, cache=True):
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']
        
        if X is None:
            X = self.X.copy()
        
        Z1 = X.dot(W1) + b1
        A1 = self.Elu(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = self.sigmoid(Z2)
        probs = A2
        
        if cache:
            self.cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 
                          'A2': A2, 'probs': probs} 
        else:
            return probs
    
    def back_prop(self):
        # Import parameters and cached values
        A1 = self.cache['A1']
        A2 = self.cache['A2']
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']
        
        # Calculate derivatives
        m = 1 / self.n_inputs
        dZ2 = A2 - self.Y.reshape(-1,1)
        dW2 = m * A1.T.dot(dZ2)
        db2 = m* np.sum(dZ2, axis=0, keepdims=True)
        dZ1 = m * dZ2.dot(W2.T) * self.dElu(A1)
        dW1 = m * np.dot(self.X.T, dZ1)
        db1 = m * np.sum(dZ1, axis=0)
        
        # Apply gradient descent updates
        W1 -= self.learning_rate * dW1
        b1 -= self.learning_rate * db1
        W2 -= self.learning_rate * dW2
        b2 -= self.learning_rate * db2
        
        # Store updated network parameters
        self.params = {'W1': W1, 'b1': b1, 
                       'W2': W2, 'b2': b2}
        
    def train(self, learning_rate=1e-2, 
              n_iters=10000, log_loss=False):
        self.learning_rate = learning_rate
        loss = []
        # Train the network
        for i in range(n_iters):
            self.forward_prop()
            self.back_prop()
            
            # Calculate the loss value to track progress
            if log_loss:
                loss.append(self.calculate_loss())
                
        if log_loss:
            return loss
    
    def predict(self, X):
        probs = self.forward_prop(X, cache=False)
        return np.where(probs<0.5,0,1)
    
    def calculate_loss(self):
        probs = self.cache['probs']
        W1 = self.params['W1']
        W2 = self.params['W2']
        Y = self.Y.reshape(-1,1)
        loss = (np.multiply(np.log(probs), Y) + 
                np.multiply(np.log(1 - probs), (1 - Y)))
        return -1 / self.n_inputs * np.sum(loss)
    
    def train_accuracy(self):
        probs = self.cache['probs']
        clf = np.where(probs<0.5, 0, 1)
        return np.sum(self.Y.reshape(-1,1)==clf) / self.n_inputs
    
    # Call this function to view the decision boundary
    def plot_decision_boundary(self):
        
        # Determine grid range in x and y directions
        x_min, x_max = self.X[:, 0].min()-0.1, self.X[:, 0].max()+0.1
        y_min, y_max = self.X[:, 1].min()-0.1, self.X[:, 1].max()+0.1
   
        # Set grid spacing parameter
        spacing = min(x_max - x_min, y_max - y_min) / 100

        # Create grid
        XX, YY = np.meshgrid(np.arange(x_min, x_max, spacing),
                       np.arange(y_min, y_max, spacing))
    
        # Concatenate data to match input
        data = np.hstack((XX.ravel().reshape(-1,1), 
                          YY.ravel().reshape(-1,1)))

        # Pass data to predict method
        clf = self.predict(data)
    
        Z = clf.reshape(XX.shape)
        
        plt.figure(figsize=(10,8))
        plt.contourf(XX, YY, Z, cmap=plt.cm.Spectral, alpha=0.8)
        plt.scatter(self.X[:,0], self.X[:,1], c=self.Y, 
                    cmap=plt.cm.cividis, s=50)
        plt.show()