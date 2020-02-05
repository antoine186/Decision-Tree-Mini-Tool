# Decision Tree Tool

## Overview

This is a function implementing a classification decision tree learning method to the training data.

(Note) This function requires Python 3.6 or higher. This tool is released with the required dependencies found in the venv folder.

## How to Use
### Exhaustive Feature Pairing

In the exhaustive pairing mode, the function will build a tree for each possible pairing of features/attributes found in your training data. Let us run through an example:

```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
```

We load the data and separate it into a training set and a label set:

```
iris = load_iris()
X = iris.data
Y = iris.target
```

We then record the number of possible classes and their corresponding plot colors:

```
feat_names = iris.feature_names
nb_class = 3
cols = "bry"
```

Finally, the function can be applied with the following command:

```
DT_Func(X, feat_names, Y, "entropy", nb_class, cols, test_size = 0.2, exhaust = True)
```

Note: Below is a different version of this function with the cross validation operation activated.

```
DT_Func(X, feat_names, Y, "entropy", nb_class, cols, test_size = 0.2, exhaust = True, k_fold = 5)
```

### Selective Feature Pairing

In the selective pairing mode, the function will build a tree for each pairing designated by the user:

```
feat1 = np.array([1,3,5,8,5])
feat2 = np.array([2,9,6,4,0])

feat_pairs = np.concatenate((feat1.reshape(5,1), feat2.reshape(5,1)), axis = 1)
```

Let us run through another example:

```
from sklearn.datasets import load_breast_cancer

bc = load_breast_cancer()
bc_feat_names = bc.feature_names
X = bc.data
Y = bc.target
nb_class = 2
cols = "br"

DT_Func(X, bc_feat_names, Y, "entropy", nb_class, cols, test_size = 0.2, feat_pairs = feat_pairs)
```

Note: Below is a different version of this function with the cross validation operation activated.

```

DT_Func(X, bc_feat_names, Y, "entropy", nb_class, cols, test_size = 0.2, feat_pairs = feat_pairs, k_fold = 5)
```

### Using all Attributes

When using all the attributes, a decision tree will be built using all of the available features in the training set.

```
DT_Complete_Func(X, bc_feat_names, Y, "entropy", nb_class, cols, test_size = 0.2, dichom = True)
```

Note: Below is a different version of this function with the cross validation operation activated.

```
DT_Complete_Func(X, bc_feat_names, Y, "entropy", nb_class, cols, test_size = 0.2, k_fold=10, dichom = True)
```

### Using our own Custom Cross-Validation

```
mod = DecisionTreeClassifier(criterion = "entropy")
res = smart_custom_fold(mod, X, Y, 5)
```

We can run the alternative command below if we are dealing with only 2 output categories and would like more performance information:

```
mod = DecisionTreeClassifier(criterion = "entropy")
res = smart_custom_fold(mod, X, Y, 5, dichom = True)
```

### Using our own Custom Confusion Matrix Function

We could use a K-nearest neighbour classifier model with 2 possible outputs and pass that to our confusion matrix function:

```
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                        X, Y, test_size=0.2, random_state=None)

my_neigh = KNeighborsClassifier(n_neighbors=5)
my_neigh = my_neigh.fit(X_train, y_train)

conf_res = compute_confuse(my_neigh, X_test, y_test, nb_class, dichom = True)
```

It is also possible to use the function in a probabilistic manner:

```
conf_res = compute_confuse(my_neigh, X_test, y_test, nb_class, dichom = True, proba = True, thresh = 0.5)
```

### Using our own Custom ROC Curve Plotter

We can plot a ROC curve to evaluate the performance of our K-nearest neighbour classifier model:

```
ROC_comp(my_neigh, X_test, y_test, nb_class, roc_steps = 100)
```

There is an alternative command, which allows us to plot a curve instead of points (Note: do not run this command straight after the previous command; This seems to cause strange problems):

```
ROC_comp(my_neigh, X_test, y_test, nb_class, roc_steps = 100, smooth_factor = 15, spline = True)
```






