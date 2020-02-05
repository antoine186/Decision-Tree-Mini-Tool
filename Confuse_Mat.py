import numpy as np

# This is a custom confusion matrix function. It can handle cases with multiple labels and cases with only true/false
# labels. For the latter, it can also compute the recall, precision, and f1 score.

# For dichom = True, please ensure that your labels are converted to 0 and 1s. 0 being a negative and 1 being a positive.
def compute_confuse(mod, test_dt, label_dt, nb_class, dichom = False, proba = False, thresh = 0.5, roc_comp = False):

    confuse_mat = np.zeros((nb_class, nb_class))

    unique_class = np.unique(label_dt)

    if (dichom == True):

        if (unique_class[0] != 0):
            raise Exception('Your target data is not in the right form of 0s and 1s.')

        if (len(unique_class) > 2):
            raise Exception('Your target data is not in the right form. It is too long.')

        false_where = np.where(unique_class == 0)
        false_where = false_where[0][0]
        true_where = 1 - false_where

    for i in range(len(test_dt)):

        if (dichom == True):

            if (proba == False):

                cur_pred = mod.predict(np.array(test_dt[i, :].reshape(1, -1)))

            else:

                if (thresh > 1 or thresh < 0):

                    raise Exception("Inappropriate threshold value")

                cur_preds = mod.predict_proba(np.array(test_dt[i, :].reshape(1, -1)))

                if (thresh < 0.5):

                    if (cur_preds[0, true_where] >= thresh):
                        cur_pred = 1

                    else:
                        cur_pred = 0

                else:

                    if (cur_preds[0, true_where] > thresh):
                        cur_pred = 1

                    else:
                        cur_pred = 0

        else:

            cur_pred = mod.predict(np.array(test_dt[i, :].reshape(1, -1)))

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

        if ((confuse_mat[true_where, true_where] + confuse_mat[true_where, false_where]) == 0):

            precision_score = 0
            recall_score = 0

            f1_score = 0

        elif((confuse_mat[true_where, true_where] + confuse_mat[false_where, true_where]) == 0):

            precision_score = 0
            recall_score = 0

            f1_score = 0

        else:

            precision_score = confuse_mat[true_where, true_where] / (
                        confuse_mat[true_where, true_where] + confuse_mat[true_where, false_where])
            recall_score = confuse_mat[true_where, true_where] / (
                        confuse_mat[true_where, true_where] + confuse_mat[false_where, true_where])

            f1_score = 2 * ((precision_score * recall_score) / (precision_score + recall_score))

        if (roc_comp == False):

            return confuse_mat, unique_class, precision_score, recall_score, f1_score

        else:

            true_pos = confuse_mat[true_where, true_where]
            false_pos = confuse_mat[true_where, false_where]
            true_neg = confuse_mat[false_where, false_where]
            false_neg = confuse_mat[false_where, true_where]

            return true_pos, false_pos, true_neg, false_neg

    else:

        return confuse_mat, unique_class




