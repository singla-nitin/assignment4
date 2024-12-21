import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
np.random.seed(42)
n_samples = 1000
X_base = np.random.rand(n_samples, 1)
X = np.hstack([X_base + np.random.normal(0, 0.01, size=(n_samples, 1)) for _ in range(7)])
y = 5 * X_base.squeeze() + np.random.normal(0, 0.1, size=n_samples)
df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(7)])
df['target'] = y
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df['target'], test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

class RidgeRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000, reg_param=0.0):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.reg_param = reg_param

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.theta = np.zeros(self.n)
        self.bias = 0

        for _ in range(self.n_iterations):
            y_pred = self.predict(X)
            error = y - y_pred

            d_theta = (-2 / self.m) * (X.T.dot(error)) + 2 * self.reg_param * self.theta
            d_bias = (-2 / self.m) * np.sum(error)

            self.theta -= self.learning_rate * d_theta
            self.bias -= self.learning_rate * d_bias

            if np.isnan(self.theta).any() or np.isnan(self.bias):
                print("NaN detected in parameters. Stopping training.")
                break

    def predict(self, X):
        return X.dot(self.theta) + self.bias

    def cost_function(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2) + self.reg_param * np.sum(self.theta ** 2)

learning_rates = [0.0001, 0.001, 0.01, 0.1]
reg_params = [1e-5, 1e-3, 0, 1]

best_cost = float('inf')
best_r2 = float('-inf')
best_lr = None
best_reg = None

for lr in learning_rates:
    for reg in reg_params:
        ridge_gd = RidgeRegressionGD(learning_rate=lr, n_iterations=1000, reg_param=reg)
        ridge_gd.fit(X_train_scaled, y_train)
        y_pred_train = ridge_gd.predict(X_train_scaled)
        y_pred_test = ridge_gd.predict(X_test_scaled)

        cost = ridge_gd.cost_function(y_train, y_pred_train)
        r2 = r2_score(y_test, y_pred_test)

        if cost < best_cost and r2 > best_r2:
            best_cost = cost
            best_r2 = r2
            best_lr = lr
            best_reg = reg

print(f"Best Learning Rate: {best_lr}")
print(f"Best Regularization Parameter: {best_reg}")
print(f"Minimum Cost: {best_cost}")
print(f"Maximum R2 Score: {best_r2}")
