from sklearn.model_selection import cross_val_score, KFold  # cross_val_score helps evaluate model performance using cross-validation. # KFold splits the data into defined folds.
from sklearn.svm import SVC # SVC is used for Support Vector Classification.
from sklearn.datasets import load_iris # load_iris loads the sample dataset.

"""
We will use the Iris dataset a built-in, multi-class dataset with 150 samples and 3 flower species (Setosa, Versicolor and Virginica).
"""

iris = load_iris()
X, y = iris.data, iris.target

'''
iris.data contains the features, and iris.target contains the labels.
x is the feature matrix, and y is the target vector.
'''

# SVC() from scikit-learn is used to build the Support Vector Machine model. Here, we are using a linear kernel, suitable for linearly separable data
svm_classifier = SVC(kernel='linear')

"""
We define 5 folds, meaning the dataset will be split into 5 parts. The model will train on 4 parts and test on 1, repeating this process 5 times for balanced evaluation.
"""
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# We use cross_val_score() to automatically split data, train and evaluate the model across all folds. It returns the accuracy for each fold
cross_val_results = cross_val_score(svm_classifier, X, y, cv=kf)

"""
We print individual fold accuracies and the mean accuracy across all folds to understand the model’s stability and generalization.
"""
print("Cross-Validation Results (Accuracy):")
for i, result in enumerate(cross_val_results, 1):
    print(f"  Fold {i}: {result * 100:.2f}%")
    
print(f'Mean Accuracy: {cross_val_results.mean()* 100:.2f}%')

## visualize the things....