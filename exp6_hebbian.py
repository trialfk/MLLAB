import numpy as np
import matplotlib.pyplot as plt

# Inputs (x1, x2) and Target (y)
x = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
y = np.array([1, -1, -1, -1])

# 2. Initialize weights and bias to zero
w1, w2, b = 0, 0, 0
print("Initial Weights: w1=0, w2=0, b=0")
print("-" * 30)

# 3. Hebbian Learning
for i in range(len(x)):
    w1 = w1 + x[i][0] * y[i]
    w2 = w2 + x[i][1] * y[i]
    b = b + y[i]
    print(f"Step {i+1}: w1={w1}, w2={w2}, b={b}")

# 4. Final Output
print("-" * 30)
print(f"Final Weights: w1={w1}, w2={w2}, b={b}")

# 5. Visualization
plt.scatter(x[:,0], x[:,1], c=y, s=100, cmap='bwr', edgecolors='k')
lims = np.array([-2, 2])
plt.plot(lims, -(w1 * lims + b) / w2, 'k-')
plt.title("Hebbian Boundary")
plt.show()
