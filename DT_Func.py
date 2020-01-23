import sklearn
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# This classifying decision tree is meant for categorical target data

# Features must be the columns of the training data and the target must be a vector array containing as many instances as
# there are rows in the training data

def DT_Func(train_dt, feat_names, label_dt, crit, nb_class, cols, feat_pairs = np.array([]), exhaust = False, plot_step = 0.02):

    # Count number of features present
    train_shape = train_dt.shape

    if exhaust == True:

        nb_features = train_shape[1]

        # Looping through all required feature pairs
        for i in range(0, nb_features):
            for j in range(i + 1, nb_features):

                # Create figure frame
                plt.figure()
                plt.rc('xtick', labelsize=8)
                plt.rc('ytick', labelsize=8)

                # Packing indices of the next feature pairs
                pair = [i, j]

                X = train_dt[:, pair]
                y = label_dt

                # Train a decision tree classifier using user specified criterion
                clf = DecisionTreeClassifier(criterion=crit).fit(X, y)

                # Plot the learned decision boundaries
                # plt.subplot(nb_features, nb_features, j * nb_features + i + 1)
                plt.plot()
                x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
                y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

                # Find classes/depths of all possible background coordinates
                xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
                Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)

                # Plot depths
                cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
                plt.xlabel(feat_names[pair[0]], fontsize=8)
                plt.ylabel(feat_names[pair[1]], fontsize=8)
                plt.axis("tight")

                # Superimpose the class colors of the training data according to their corresponding targets
                for ii, color in zip(range(nb_class), cols):
                    idx = np.where(y == ii)
                    plt.scatter(X[idx, 0], X[idx, 1], c=color, cmap=plt.cm.Paired)
                plt.axis("tight")

        plt.show()

    elif exhaust == False:

        #nb_features = len(feat_pairs) * 2
        nb_features = len(feat_pairs)

        for q in range(len(feat_pairs)):

            # Create figure frame
            plt.figure()
            plt.rc('xtick', labelsize=8)
            plt.rc('ytick', labelsize=8)

            i = feat_pairs[q, 0]
            j = feat_pairs[q, 1]

            pair = feat_pairs[q]

            X = train_dt[:, pair]
            y = label_dt

            # Train a decision tree classifier using user specified criterion
            clf = DecisionTreeClassifier(criterion=crit).fit(X, y)

            # Plot the learned decision boundaries
            plt.plot()
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

            # Find classes/depths of all possible background coordinates
            xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            # Plot depths
            cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
            plt.xlabel(feat_names[pair[0]], fontsize=8)
            plt.ylabel(feat_names[pair[1]], fontsize=8)
            plt.axis("tight")

            # Superimpose the class colors of the training data according to their corresponding targets
            for ii, color in zip(range(nb_class), cols):
                idx = np.where(y == ii)
                plt.scatter(X[idx, 0], X[idx, 1], c=color, cmap=plt.cm.Paired)
            plt.axis("tight")

        plt.show()



