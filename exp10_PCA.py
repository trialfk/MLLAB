import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# 1. Load a 4D dataset (Iris has 4 features)
data = load_iris()
X = data.data
y = data.target

# 2. Initialize and apply PCA to reduce 4D to 2D
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# 3. Output Metrics
print("Original shape:", X.shape)
print("Reduced shape:", X_reduced.shape)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# 4. Simple Visualization
plt.scatter(X_reduced[:, 0], X_reduced[:, 1],
             c=y, cmap='viridis', edgecolors='k')
plt.title("PCA: 4D reduced to 2D")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()