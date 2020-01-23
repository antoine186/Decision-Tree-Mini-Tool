import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer

bc = load_breast_cancer()
bc_feat_names = bc.feature_names
X = bc.data
Y = bc.target

DT_Func(X, bc_feat_names, Y, 3, "entropy", 3, )

# This classifying decision tree is meant for binary target data

# Features must be the columns of the training data and the target must be a vector array containing as many instances as
# there are rows in the training data

# nb_combo is the number of different pairs of features one would like to try

# crit is the criterion for the decision tree
def DT_Func(train_dt, feat_names, label_dt, nb_combo, crit, nb_class, cols,exhaust = False, plot_step = 0.02):

    nb_features = train_dt.shape

    plt.figure()
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    for i in range(0, nb_features):
        for j in range(i + 1, nb_features):

            # classify using two corresponding features
            pair = [i, j]
            X = train_dt[:, pair]
            y = label_dt
            # train classifier
            clf = DecisionTreeClassifier(criterion = crit).fit(X, y)
            # plot the (learned) decision boundaries
            plt.subplot(nb_features, nb_features, j * nb_features + i + 1)
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

            xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
            plt.xlabel(feat_names[pair[0]], fontsize=8)
            plt.ylabel(feat_names[pair[1]], fontsize=8)
            plt.axis("tight")
            # plot the training points
            for ii, color in zip(range(nb_class), cols):
                print(color)
                idx = np.where(y == ii)
                plt.scatter(X[idx, 0], X[idx, 1], c=color, cmap=plt.cm.Paired)
            plt.axis("tight")

    plt.show()
