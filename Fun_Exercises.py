from DT_Func import DT_Func

###### Exercise 1

### We will fit 10 trees to breast Cancer data using all the available features

from sklearn.datasets import load_breast_cancer

bc = load_breast_cancer()
bc_feat_names = bc.feature_names
X = bc.data
Y = bc.target
nb_class = 2
cols = "br"

DT_Func(X, bc_feat_names, Y, "entropy", nb_class, cols, test_size = 0.2, exhaust = True)