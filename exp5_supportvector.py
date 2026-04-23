import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Generate dataset
X, y = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.5)

# 2. Train model
model = SVC(kernel='linear')
model.fit(X, y)

# 3. Predict
y_pred = model.predict(X)

# 4. Metrics
print("Accuracy:", accuracy_score(y, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y, y_pred))
print("\nReport:\n", classification_report(y, y_pred))

# 5. Improved Plot
plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr', alpha=0.6)
xx, yy = np.meshgrid(
    np.linspace(X[:,0].min()-1, X[:,0].max()+1, 50),
    np.linspace(X[:,1].min()-1, X[:,1].max()+1, 50)
)
Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.contour(xx, yy, Z, levels=[0])           
plt.contour(xx, yy, Z, levels=[-1, 1], linestyles='dashed') 
plt.title("Support Vector Machine ")
plt.show()
