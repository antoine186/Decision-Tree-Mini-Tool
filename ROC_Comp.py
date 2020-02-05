import numpy as np
import io
import sys
from Confuse_Mat import compute_confuse
from Simple_Scatter import simple_scatter

def ROC_comp(mod, test_dt, label_dt, nb_class, roc_steps, smooth_factor = 15, spline = False):

    roc_rates = np.zeros((roc_steps + 1, 2))

    # This output trapping code has been found on codingdose.info at the following link:
    # https://codingdose.info/2018/03/22/supress-print-output-in-python/
    # Answer was provided by username FRANCCESCO OROZCO
    text_trap = io.StringIO()
    sys.stdout = text_trap

    thresh_incr = 1 / roc_steps
    #thresh_step = thresh_incr
    thresh_step = 0

    for i in range(roc_steps + 1):

        conf_res = compute_confuse(mod, test_dt, label_dt, nb_class, dichom=True, proba=True, thresh=thresh_step, roc_comp=True)

        if (i == roc_steps):
            thresh_step = 1

        else:
            thresh_step = thresh_step + thresh_incr

        if (thresh_step > 1):
            thresh_step = 1

        roc_rates[i, 0] = (conf_res[0] / (conf_res[0] + conf_res[3])) * 100

        roc_rates[i, 1] = (conf_res[1] / (conf_res[1] + conf_res[2])) * 100

    sys.stdout = sys.__stdout__

    simple_scatter(roc_rates, "True Positive Rate (Sensitivity)", "False Positive Rate (100 - Specificity)", "ROC Curve",
                   smooth_factor = smooth_factor, spline = spline)

    return
