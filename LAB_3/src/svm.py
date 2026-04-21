import numpy as np
from tqdm import tqdm

class SVM:
    def __init__(self, learning_rate=0.001, C=1.0, epochs=50):
        self.lr = learning_rate
        self.C = C
        self.epochs = epochs
        self.theta = None
        self.b = 0.0
        self.loss_history = []

    def hinge_loss(self, y, y_hat):
        distance = 1 - y * y_hat
        hinge_loss = self.C * np.where(distance > 0, distance, 0).sum()
        l2_loss = 0.5 * (self.theta.T @ self.theta)
        return (l2_loss + hinge_loss) / y.size

    def fit(self, X, y):
        N = X.shape[0]
        self.theta = np.zeros(X.shape[1])
        self.b = 0.0
        
        progress_bar = tqdm(range(self.epochs), desc="Training SVM (SGD)")
        np.random.seed(42)
        for epoch in progress_bar:
            indices = np.arange(N)
            np.random.shuffle(indices)
            
            for i in indices:
                x_i = X[i]
                y_i = y[i]

                condition = y_i * (np.dot(x_i, self.theta) + self.b) >= 1
                
                if condition:
                    gradient_theta = self.theta / N
                    gradient_b = 0
                else:
                    gradient_theta = (self.theta / N) - self.C * y_i * x_i
                    gradient_b = -self.C * y_i
                
                self.theta -= self.lr * gradient_theta
                self.b -= self.lr * gradient_b
            
            y_hat_all = X @ self.theta + self.b
            loss = self.hinge_loss(y, y_hat_all)
            self.loss_history.append(loss)
            
            progress_bar.set_postfix({"Loss": f"{loss:.4f}"})

    def predict(self, X):
        """Dự đoán nhãn là -1 hoặc 1"""
        predictions = np.sign(X @ self.theta + self.b)
        predictions[predictions == 0] = 1
        return predictions