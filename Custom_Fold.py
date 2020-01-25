from math import floor
import numpy as np

# This function is the implementation of a custom cross-validation operation. It does not work the best, and that is because
# only the base case has been implemented and the folds are not randomly sampled
def custom_fold(mod, train_dt, label_dt, k_fold):

    train_len = len(train_dt)
    fold_len = floor(train_len/k_fold)

    # Base case
    if ((train_len % fold_len) == 0):

        fold_bounds = np.zeros(k_fold + 1)
        fold_bounds[0] = -1

        for i in range(1, k_fold + 1):
            fold_bounds[i] = fold_bounds[i-1] + fold_len

        fold_bounds[0] = 0

    else:
        print("hi")

    bound_ind = 0
    fold_bounds = fold_bounds.astype(int)

    score_contain = np.zeros(k_fold)

    for i in range(k_fold):
        cur_testfold = train_dt[fold_bounds[bound_ind]:fold_bounds[bound_ind+1], :]
        cur_testlabel = label_dt[fold_bounds[bound_ind]:fold_bounds[bound_ind+1]]

        cur_trainfold = np.delete(train_dt, np.asarray(list(range(fold_bounds[bound_ind], fold_bounds[bound_ind+1] + 1))), axis = 0)
        cur_trainlabel = np.delete(label_dt, np.asarray(list(range(fold_bounds[bound_ind], fold_bounds[bound_ind+1] + 1))), axis = 0)

        bound_ind = bound_ind + 1

        cur_mod = mod.fit(cur_trainfold, cur_trainlabel)
        score_contain[i] = cur_mod.score(cur_testfold, cur_testlabel)

    return(score_contain)

