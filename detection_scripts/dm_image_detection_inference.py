#
# Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt
#

import os
import numpy as np
import pandas
from sklearn import metrics
import dmetrics
import argparse
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score, roc_auc_score, recall_score


def calculate_metrics(csv_path,op_path,output_path):
    os.makedirs(output_path, exist_ok=True)
    dist_metrics = {
        'acc': lambda y_label, y_pred: dmetrics.balanced_accuracy_score(y_label, y_pred > 0),
        'pd10': lambda y_label, y_pred: dmetrics.pd_at_far(y_label, y_pred, 0.10),
        'eer': lambda y_label, y_pred: dmetrics.calculate_eer(y_label, y_pred),
        'pd05': lambda y_label, y_pred: dmetrics.pd_at_far(y_label, y_pred, 0.05),
        'pd01': lambda y_label, y_pred: dmetrics.pd_at_far(y_label, y_pred, 0.01),
        'macc': lambda y_label, y_pred: dmetrics.macc(y_label, y_pred),
        'count1': lambda y_label, y_pred: np.sum(y_label == 1),
        'count0': lambda y_label, y_pred: np.sum(y_label == 0),
        
        'acc_total': lambda y_label, y_pred: accuracy_score(y_label, y_pred > 0.5),
        'ap': lambda y_label, y_pred: average_precision_score(y_label, y_pred),
        'recall': lambda y_label, y_pred:  recall_score(y_label, y_pred > 0.5)
    }

    # Changed here
    df=pd.read_csv(op_path)
    db0s = list(set(df['typ'])) 
    db1d = {}
    for val in db0s: db1d[val]=[str(val)]

    # NOTE: all the methodologies to use to evaluate the metrics for
    mm = ['Grag2021_progan', 'Grag2021_latent']
    tab_metrics_p1 = pandas.DataFrame(index=db1d.keys(), columns=mm)
    tab_metrics_p2 = pandas.DataFrame(index=db1d.keys(), columns=mm)
    tab_metrics_p3 = pandas.DataFrame(index=db1d.keys(), columns=mm)

    


    tab_rs = []
    for db0 in db0s:
        tab_rs.append(pandas.read_csv(os.path.join(os.path.join(csv_path, db0), db0 + ".csv"), index_col='src'))
    tab_r = pandas.concat(tab_rs)
    for db1 in db1d:
        tab_f = []
        for folder in db1d[db1]:
            tab_f.append(pandas.read_csv(os.path.join(os.path.join(csv_path, folder), folder + ".csv")))
        if len(tab_f) > 1:
            tab_f = pandas.concat(tab_f)
        else:
            tab_f = tab_f[0]
        tab_all = []
        tab_all.append(tab_f)
        tab_all.append(tab_r)
        tab_both = pandas.concat(tab_all)
        label = tab_both['label']
        for method in mm:
            predict = tab_both[method]
            v = predict[np.isfinite(predict)]
            predict = predict.clip(np.min(v), np.max(v))
            predict[np.isnan(predict)] = 0.0
            tab_metrics_p1.loc[db1, method] = dist_metrics['acc_total'](label, predict)
            tab_metrics_p2.loc[db1, method] = dist_metrics['ap'](label, predict)
            tab_metrics_p3.loc[db1, method] = dist_metrics['recall'](label, predict)




    tab_metrics_p1.loc['AVR'] = tab_metrics_p1.mean(0)
    tab_metrics_p2.loc['AVR'] = tab_metrics_p2.mean(0)
    tab_metrics_p3.loc['AVR'] = tab_metrics_p3.mean(0)



    r_acc_path = os.path.join(output_path, "acc_total.csv")
    tab_metrics_p1.to_csv(r_acc_path)
    
    f_acc_path = os.path.join(output_path, "ap.csv")
    tab_metrics_p2.to_csv(f_acc_path)
    
    acc_total_path = os.path.join(output_path, "recall.csv")
    tab_metrics_p3.to_csv(acc_total_path)
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=str, help="The path to of the csv files of the output of the networks", default="./results_tst")
    parser.add_argument("--op_dir", type=str, help="The path of the operations file")
    parser.add_argument("--out_dir", type=str, help="The Path where the csv containing the calculated metrics of the networks should be saved", default="./")
    args = vars(parser.parse_args())
    calculate_metrics(args['csv_dir'], args['op_dir'],args['out_dir'])


main()
