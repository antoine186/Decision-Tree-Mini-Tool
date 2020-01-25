# Decision Tree Mini Tool

## Overview

This is a function implementing a classification decision tree learning method to the training data.

(Note) This function requires python 3.6 or higher. This tool is released with the required dependencies found in the venv folder.

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
DT_Complete_Func(X, bc_feat_names, Y, "entropy", nb_class, cols, test_size = 0.2)
```

Note: Below is a different version of this function with the cross validation operation activated.

```
DT_Complete_Func(X, bc_feat_names, Y, "entropy", nb_class, cols, test_size = 0.2, k_fold=10)
```

### Using our own Custom Cross-Validation

```
mod = DecisionTreeClassifier(criterion = "entropy")
res = smart_custom_fold(mod, X, Y, 5)
```
