import matplotlib.pyplot as plt

# This is a simple function for the generation of a scatter plot.

# For N points, it takes a Nx2 numpy array as input. The first column contains the y-vals and the second the x-vals.
def simple_scatter(coords, y_lab, x_lab, plot_title):

    plt.scatter(coords[:,1], coords[:,0])

    plt.title(plot_title)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.show()



