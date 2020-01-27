import numpy as np

def compute_confuse(mod, test_dt, label_dt, nb_class, dichom = False):

    confuse_mat = np.zeros((nb_class, nb_class))

    unique_class = np.unique(label_dt)

    for i in range(len(test_dt)):
        cur_pred = mod.predict(np.array(test_dt[i,:].reshape(1, -1)))

        if (cur_pred == label_dt[i]):

            cur_ind_tuple = np.where(unique_class == label_dt[i])

            confuse_mat[cur_ind_tuple[0][0], cur_ind_tuple[0][0]] = confuse_mat[cur_ind_tuple[0][0], cur_ind_tuple[0][0]] + 1

        else:

            true_ind_tuple = np.where(unique_class == label_dt[i])
            false_ind_tuple = np.where(unique_class == cur_pred)

            confuse_mat[false_ind_tuple[0][0], true_ind_tuple[0][0]] = confuse_mat[false_ind_tuple[0][0], true_ind_tuple[0][0]] + 1

    print("Confusion Matrix:")

    # This string formatting code has been found on stackoverflow.com at the following link:
    # https://stackoverflow.com/questions/9535954/printing-lists-as-tabular-data
    # Answer was provided by username Sven Marnach

    ax_len = len(unique_class) + 2
    top_ax = np.repeat("", ax_len)

    row_format = "{:>15}" * (len(unique_class) + 2)
    print(row_format.format("", "", *unique_class))
    for ans, row in zip(unique_class, confuse_mat):
        print(row_format.format("", ans, *row))

    print("Note: The columns represent actual labels while the rows represent predicted labels.")

    if (dichom == True):

        false_where = np.where(unique_class == 0)
        false_where = false_where[0][0]
        true_where = 1 - false_where

        precision_score = confuse_mat[true_where, true_where] / (confuse_mat[true_where, true_where] + confuse_mat[true_where, false_where])
        recall_score = confuse_mat[true_where, true_where] / (confuse_mat[true_where, true_where] + confuse_mat[false_where, true_where])

        f1_score = 2 * ((precision_score * recall_score) / (precision_score + recall_score))

        return confuse_mat, unique_class, precision_score, recall_score, f1_score

    else:

        return confuse_mat, unique_class




