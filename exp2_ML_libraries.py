# NumPy → numerical operations
import numpy as np
a = np.array([1, 2, 3, 4, 5])
print("--- NumPy ---")
print("Array:", a)
print("Mean:", np.mean(a))

# Pandas → data manipulation (DataFrames)
import pandas as pd
data = {'Name': ['A', 'B', 'C'], 'Marks': [85, 90, 88]}
df = pd.DataFrame(data)
print("\n--- Pandas ---")
print(df)

# SciPy → scientific/statistical functions
from scipy import stats
data = [10, 20, 30, 40, 50]
print("\n--- SciPy ---")
print("Mean:", stats.tmean(data))

# Matplotlib/Seaborn → visualization
import matplotlib.pyplot as plt
x = [1, 2, 3, 4]
y = [10, 20, 25, 30]
print("\n--- Matplotlib ---")
plt.plot(x, y)
plt.xlabel("x-axis")
plt.ylabel("Y-axis")
plt.title("Simple Line Plot")
plt.show()

import seaborn as sns
import pandas as pd
data = pd.DataFrame({'x': [1, 2, 3, 4], 'y': [10, 20, 25, 30]})
print("\n--- Seaborn ---")
sns.scatterplot(x='x', y='y', data=data)
plt.show()

# Scikit-learn → ML models
from sklearn.linear_model import LinearRegression
import numpy as np
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 6, 8, 10])
model = LinearRegression()
model.fit(X, y)
print("\n--- Scikit-learn ---")
print("Prediction for 6:", model.predict([[6]]))


import os  #Removes the warning messages , dont write on paper
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

# TensorFlow/Keras → Deep Learning
import tensorflow as tf
a = tf.constant(10)
b = tf.constant(20)
print("\n--- TensorFlow ---")
print("Sum:", tf.add(a, b).numpy())

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input
model_keras = Sequential([
    Input(shape=(4,)),
    Dense(8, activation='relu'),
    Dense(1)
])
print("\n--- Keras ---")
model_keras.summary()
