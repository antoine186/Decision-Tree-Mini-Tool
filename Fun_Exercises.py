import numpy as np
from DT_Func import DT_Func
from DT_Func import DT_Complete_Func

###### Exercise 1

### We will fit 10 trees to breast Cancer data using all the available features

from sklearn.datasets import load_breast_cancer

bc = load_breast_cancer()
bc_feat_names = bc.feature_names
X = bc.data
Y = bc.target
nb_class = 2
cols = "br"

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

print("Accuracy mean for non-cross validated decision tree fittings is " + str(non_cross_val_mean))
print("The corresponding standard error is " + str(non_cross_val_SD))