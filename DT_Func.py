import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer

bc = load_breast_cancer()
bc_feat_names =

X = bc.data[:, [1, 3]]

######
