import numpy as np

# 1. Input patterns
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# 2. Weights and threshold
w = np.array([1, 1])
threshold = 2

# 3. Activation function
def mcculloch_pitts(x):
    return 1 if np.dot(x, w) >= threshold else 0

# 4. Output
print("Input -> Output")
for i in X:
    print(i, "->", mcculloch_pitts(i))