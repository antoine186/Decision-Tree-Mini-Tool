from math import floor
from random import sample
import numpy as np

def smart_custom_fold(mod, train_dt, label_dt, k_fold):

    main_mod = mod

    train_len = len(train_dt)
    fold_len = floor(train_len/k_fold)

    # Base case
    if ((train_len % fold_len) == 0):

        sample_list = list(range(0, train_len))
        global_indices = sample(sample_list, train_len)

        fold_bounds = np.zeros(k_fold + 1)
        fold_bounds[0] = -1

        for i in range(1, k_fold + 1):
            fold_bounds[i] = fold_bounds[i - 1] + fold_len

        fold_bounds[0] = 0

    else:

        print("Under construction")

    bound_ind = 0
    fold_bounds = fold_bounds.astype(int)

    score_contain = np.zeros(k_fold)

    for i in range(k_fold):
        cur_test_indices = np.asarray(global_indices[fold_bounds[bound_ind]:fold_bounds[bound_ind + 1]])
        cur_train_indices = np.delete(np.asarray(global_indices), global_indices[fold_bounds[bound_ind]:fold_bounds[bound_ind + 1]])

        cur_testfold = train_dt[cur_test_indices, :]
        cur_testlabel = label_dt[cur_test_indices]

        cur_trainfold = train_dt[cur_train_indices, :]
        cur_trainlabel = label_dt[cur_train_indices]

        bound_ind = bound_ind + 1

        mod = main_mod

        mod.fit(cur_trainfold, cur_trainlabel)
        score_contain[i] = mod.score(cur_testfold, cur_testlabel)

    return (score_contain)

