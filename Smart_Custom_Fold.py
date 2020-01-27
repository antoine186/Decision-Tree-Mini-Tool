from math import floor
from random import sample
import numpy as np
from Confuse_Mat import compute_confuse

def smart_custom_fold(mod, train_dt, label_dt, k_fold, dichom = False):

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

        raise Exception('Your input data of length {} is not divisible into equally sized {} folds'.format(len(train_dt), k_fold))

    bound_ind = 0
    fold_bounds = fold_bounds.astype(int)

    score_contain = np.zeros(k_fold)
    metrics_contain = np.zeros((k_fold, 3))

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

        if (dichom == True):

            confuse_mat, unique_classes, precision_score, recall_score, f1_score \
                = compute_confuse(mod, cur_testfold, cur_testlabel, nb_class = 2, dichom=dichom)

            metrics_contain[i, 0] = precision_score
            metrics_contain[i, 1] = recall_score
            metrics_contain[i, 2] = f1_score

    if (dichom == True):
        print("Mean precision across all folds is: " + str(np.mean(metrics_contain[:, 0])))
        print("SD of precision across all folds is: " + str(np.std(metrics_contain[:, 0])))
        print("Mean recall across all folds is: " + str(np.mean(metrics_contain[:, 1])))
        print("SD of recall across all folds is: " + str(np.std(metrics_contain[:, 1])))
        print("Mean f1 score across all folds is: " + str(np.mean(metrics_contain[:, 2])))
        print("SD of f1 score across all folds is: " + str(np.std(metrics_contain[:, 2])))

    return (score_contain)

