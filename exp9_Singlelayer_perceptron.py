import numpy as np

# 1. Input data (bipolar AND)
X = np.array([
    [1, 1],
    [1, -1],
    [-1, 1],
    [-1, -1]
])

y = np.array([1, -1, -1, -1])

# 2. Initialize
w = np.zeros(2)
b = 0
lr = 0.1
epochs = 50

# 3. Training
for epoch in range(epochs):
    errors = 0
    
    for i in range(len(X)):
        y_pred = np.sign(np.dot(X[i], w) + b)
        
        if y_pred != y[i]:
            w = w + lr * X[i] * y[i]
            b = b + lr * y[i]
            errors += 1

    # Print every 10 epochs (to avoid too much output)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs}:")
        print(f"Weights: {w}")
        print(f"Bias: {b}")
        print(f"Errors in this epoch: {errors}")
        print("-" * 40)