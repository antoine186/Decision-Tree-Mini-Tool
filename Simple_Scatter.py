import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

# This is a simple function for the generation of a scatter plot.

# For N points, it takes a Nx2 numpy array as input. The first column contains the y-vals and the second the x-vals.
def simple_scatter(coords, y_lab, x_lab, plot_title, spline = False):

    if (spline == False):

        plt.scatter(coords[:, 1], coords[:, 0])

        plt.title(plot_title)
        plt.xlabel(x_lab)
        plt.ylabel(y_lab)
        plt.show()

    else:

        x_sorted = np.sort(coords[:, 1], axis=None)
        y_sorted = np.sort(coords[:, 0], axis=None)

        tck = interpolate.splrep(x_sorted, y_sorted, s=0)
        xnew = np.arange(0, 100, 0.1)
        ynew = interpolate.splev(xnew, tck, der=0)

        plt.plot(xnew, ynew)

        plt.title(plot_title)
        plt.xlabel(x_lab)
        plt.ylabel(y_lab)
        plt.show()




