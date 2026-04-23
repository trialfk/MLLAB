import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Randomly generate a regression dataset
X, y = make_regression(n_samples=100, n_features=1, noise=15, random_state=42)

# 2. Initialize and train the model
model = LinearRegression()
model.fit(X, y)

# 3. Predict values
y_pred = model.predict(X)

# 4. Output Metrics
print(f"MAE: {mean_absolute_error(y, y_pred):.2f}")
print(f"MSE: {mean_squared_error(y, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.2f}")
print(f"R2 Score: {r2_score(y, y_pred):.2f}")

# 5. Visualization 
plt.scatter(X, y, alpha=0.5)
plt.plot(X, y_pred)
plt.title("Linear Regression")
plt.xlabel("Input Feature")
plt.ylabel("Target Value")
plt.show()
