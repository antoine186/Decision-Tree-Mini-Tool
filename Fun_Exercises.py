###### Global imports

import numpy as np
from DT_Func import DT_Func
from DT_Func import DT_Complete_Func
from Custom_Fold import custom_fold
from Smart_Custom_Fold import smart_custom_fold
from sklearn.tree import DecisionTreeClassifier
from ROC_Comp import ROC_comp
from Confuse_Mat import compute_confuse

###### Breast Cancer Import

from sklearn.datasets import load_breast_cancer

bc = load_breast_cancer()
bc_feat_names = bc.feature_names
X = bc.data
Y = bc.target
nb_class = 2
cols = "br"

feat1 = np.array([1,3,5,8,5])
feat2 = np.array([2,9,6,4,0])

feat_pairs = np.concatenate((feat1.reshape(5,1), feat2.reshape(5,1)), axis = 1)

###### Iris import

from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
Y = iris.target

feat_names = iris.feature_names
nb_class = 3
cols = "bry"

###### Exercise 1

### We will fit 10 trees to breast Cancer data using all the available features

non_cross_val_scores = np.zeros(10)

non_cross_val_scores[0] = DT_Complete_Func(X, bc_feat_names, Y, "entropy", nb_class, cols, test_size = 0.2)
non_cross_val_scores[1] = DT_Complete_Func(X, bc_feat_names, Y, "entropy", nb_class, cols, test_size = 0.2)
non_cross_val_scores[2] = DT_Complete_Func(X, bc_feat_names, Y, "entropy", nb_class, cols, test_size = 0.2)
non_cross_val_scores[3] = DT_Complete_Func(X, bc_feat_names, Y, "entropy", nb_class, cols, test_size = 0.2)
non_cross_val_scores[4] = DT_Complete_Func(X, bc_feat_names, Y, "entropy", nb_class, cols, test_size = 0.2)
non_cross_val_scores[5] = DT_Complete_Func(X, bc_feat_names, Y, "entropy", nb_class, cols, test_size = 0.2)
non_cross_val_scores[6] = DT_Complete_Func(X, bc_feat_names, Y, "entropy", nb_class, cols, test_size = 0.2)
non_cross_val_scores[7] = DT_Complete_Func(X, bc_feat_names, Y, "entropy", nb_class, cols, test_size = 0.2)
non_cross_val_scores[8] = DT_Complete_Func(X, bc_feat_names, Y, "entropy", nb_class, cols, test_size = 0.2)
non_cross_val_scores[9] = DT_Complete_Func(X, bc_feat_names, Y, "entropy", nb_class, cols, test_size = 0.2)

non_cross_val_mean = np.mean(non_cross_val_scores)
non_cross_val_SD = np.std(non_cross_val_scores)

print("Accuracy mean for decision tree fittings is " + str(non_cross_val_mean))
print("The corresponding standard error is " + str(non_cross_val_SD))

cross_val_score = DT_Complete_Func(X, bc_feat_names, Y, "entropy", nb_class, cols, test_size = 0.2, k_fold=10)

###### Exercise 2

### We will fit both a non-cross-validated and cross-validated tree to the iris dataset and observe the difference in
# generalisation performance. This is for exhaustive search

DT_Func(X, feat_names, Y, "entropy", nb_class, cols, test_size = 0.2, exhaust = True)
DT_Func(X, feat_names, Y, "entropy", nb_class, cols, test_size = 0.2, exhaust = True, k_fold = 5)

###### Exercise 3

### We will fit both a non-cross-validated and cross-validated tree to the breast Cancer dataset and observe the difference in
# generalisation performance. This is for selective pairing

DT_Func(X, bc_feat_names, Y, "entropy", nb_class, cols, test_size = 0.2, feat_pairs = feat_pairs)
DT_Func(X, bc_feat_names, Y, "entropy", nb_class, cols, test_size = 0.2, feat_pairs = feat_pairs, k_fold = 5)

###### Exercise 4

### We will run our own naive custom-built cross-validation function in order to compare results with the function from sklearn
# This exercise uses the Iris dataset

mod = DecisionTreeClassifier(criterion = "entropy")
res = custom_fold(mod, X, Y, 5)

print("Mean accuracy score for custom cross-validation tree with 5-fold(s) is: " + str(np.mean(res)))
print("Corresponding SD for custom cross-validation tree with 5-fold(s) is: " + str(np.std(res)))

print("-------")

DT_Complete_Func(X, feat_names, Y, "entropy", nb_class, cols, test_size = 0.2, k_fold=5)

###### Exercise 5

### We will run our own smart custom-built cross-validation function in order to compare results with the function from sklearn
# This exercise uses the Iris dataset

mod = DecisionTreeClassifier(criterion = "entropy")
res = smart_custom_fold(mod, X, Y, 5)

print("Mean accuracy score for custom cross-validation tree with 5-fold(s) is: " + str(np.mean(res)))
print("Corresponding SD for custom cross-validation tree with 5-fold(s) is: " + str(np.std(res)))

print("-------")

DT_Complete_Func(X, feat_names, Y, "entropy", nb_class, cols, test_size = 0.2, k_fold=5)

###### Exercise 6

# We will test our own custom-built confusion matrix function using the Cancer dataset

from Confuse_Mat import compute_confuse
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                        X, Y, test_size=0.2, random_state=None)

# Train a decision tree classifier using user specified criterion
clf = DecisionTreeClassifier(criterion="entropy").fit(X_train, y_train)

mat, classes = compute_confuse(clf, X_test, y_test, nb_class)

DT_Complete_Func(X, bc_feat_names, Y, "entropy", nb_class, cols, test_size = 0.2)

###### Exercise 7

# We will test our own custom-built confusion matrix function using the Cancer dataset within the DT_Complete_Func function
# call

DT_Complete_Func(X, bc_feat_names, Y, "entropy", nb_class, cols, test_size = 0.2)
print("--------")
DT_Complete_Func(X, bc_feat_names, Y, "entropy", nb_class, cols, test_size = 0.2, dichom = True)
print("--------")
DT_Complete_Func(X, bc_feat_names, Y, "entropy", nb_class, cols, test_size = 0.2, k_fold = 5)
print("--------")
DT_Complete_Func(X, bc_feat_names, Y, "entropy", nb_class, cols, test_size = 0.2, k_fold = 5, dichom = True)

# We will also test our own custom-built confusion matrix function using the Cancer dataset within the custom cross_val function
# call

X = X[0:565,:]

mod = DecisionTreeClassifier(criterion = "entropy")
res = smart_custom_fold(mod, X, Y, 5, dichom = True)

print("Mean accuracy score for custom cross-validation tree with 5-fold(s) is: " + str(np.mean(res)))
print("Corresponding SD for custom cross-validation tree with 5-fold(s) is: " + str(np.std(res)))

###### Exercise 8

# Attempting to train a k-nearest neighbour classifier in order to compute a ROC curve using the Cancer dataset

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                        X, Y, test_size=0.2, random_state=None)

my_neigh = KNeighborsClassifier(n_neighbors=5)
my_neigh = my_neigh.fit(X_train, y_train)

ROC_comp(my_neigh, X_test, y_test, nb_class, roc_steps = 100)
ROC_comp(my_neigh, X_test, y_test, nb_class, roc_steps = 100, smooth_factor = 15, spline = True)

###### Exercise 9

# Validating our custom cross validation method by applying them to a k-nearest neighbour classifier using the Cancer
# dataset

from sklearn.neighbors import KNeighborsClassifier

X = X[0:565,:]
Y = Y[0:565]

my_neigh = KNeighborsClassifier(n_neighbors=5)

res = smart_custom_fold(my_neigh, X, Y, 5, dichom = True)





