import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# 1. Generate a dataset with overlapping clusters
X, y = make_blobs(n_samples=300, centers=3, cluster_std=1.2, random_state=42)

# 2. Initialize and train the EM model (Gaussian Mixture)
# n_components is the number of clusters
model = GaussianMixture(n_components=3, random_state=42)
model.fit(X)

# 3. Predict the cluster for each data point
y_pred = model.predict(X)

# 4. Output Metrics
print(f"Converged: {model.converged_} | Iterations: {model.n_iter_}")
print("\nCluster Means:\n", model.means_)
print("\nCovariance Matrices:\n", model.covariances_)
print("\nCluster Weights:\n", model.weights_)
print("\n Log Likelihood Score:\n", model.score(X))

# 5. Visualization
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=30, alpha=0.6)
plt.scatter(model.means_[:, 0], model.means_[:, 1], marker='X', s=200, color='black', label='Centers')
plt.title("Expectation Maximization (GMM)")
plt.legend()
plt.show()