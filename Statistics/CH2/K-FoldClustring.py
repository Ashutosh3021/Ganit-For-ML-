from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score,KFold
from sklearn.datasets import load_iris

Data_set=load_iris()

x,y = Data_set.data , Data_set.target

classifier = SVC(kernel='linear')

k = 5
folds = KFold(n_splits=k , shuffle=True , random_state=42)

cross_res = cross_val_score(classifier , x , y , cv=folds)

print("Cross-Validation Results (Accuracy):")
for i, result in enumerate(cross_res, 1):
    print(f"  Fold {i}: {result * 100:.2f}%")
    
print(f'Mean Accuracy: {cross_res.mean()* 100:.2f}%')
