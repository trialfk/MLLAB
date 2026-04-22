import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Generate a random binary classification dataset
X, y = make_classification(n_samples=100, n_features=1, n_informative=1, 
                           n_redundant=0, n_clusters_per_class=1, random_state=42)

# 2. Initialize and train the model
model = LogisticRegression()
model.fit(X, y)

# 3. Predict values
y_pred = model.predict(X)

# 4. Output Metrics
print(f"Accuracy Score: {accuracy_score(y, y_pred):.2f}")
print("\nConfusion Matrix:\n", confusion_matrix(y, y_pred))
print("\nClassification Report:\n", classification_report(y, y_pred))

# 5. Visualization
plt.title("Logistic Regression")
plt.scatter(X, y, alpha=0.5)
x = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
plt.plot(x, model.predict_proba(x)[:, 1])
plt.xlabel("Feature Value")
plt.ylabel("Class (0 or 1)")
plt.show()