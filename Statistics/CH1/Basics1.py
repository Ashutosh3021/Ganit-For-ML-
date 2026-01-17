# 1. Load Iris dataset
from sklearn.datasets import load_iris
iris = load_iris()

# 2. Explore data
print(iris.data.shape)      # (150, 4)
print(iris.target_names)    # ['setosa' 'versicolor' 'virginica']

# 3. Visualize with matplotlib
import matplotlib.pyplot as plt
plt.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()

# 4. Numpy practice: Calculate mean of each feature
means = iris.data.mean(axis=0)  # ✓ NumPy improvement
