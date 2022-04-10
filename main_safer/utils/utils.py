import sys
import time
import warnings

import numpy as np
import torch
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")
sys.path.append("../../main")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# For printing cleaner numpy arrays
np.set_printoptions(formatter={'float_kind': lambda x: "%.3f" % x})


def calc_elapsed_time(start, end):
    hours, rem = divmod(end - start, 3600)
    time_hours, time_rem = divmod(end, 3600)
    minutes, seconds = divmod(rem, 60)
    time_mins, _ = divmod(time_rem, 60)
    return int(hours), int(minutes), seconds


def evaluation_measures(preds, labels):
    f1 = f1_score(labels, preds, average='binary', pos_label=1)
    f1_macro = f1_score(labels, preds, average='macro')
    # print(metrics.classification_report(labels, preds))
    return f1, f1_macro


def print_stats(max_epoch, epoch, train_loss, train_f1, train_f1_macro, val_loss, val_f1, val_f1_macro, start, lr):
    end = time.time()
    hours, minutes, seconds = calc_elapsed_time(start, end)

    train_loss = sum(train_loss) / len(train_loss)
    print("\nEpoch: {}/{},  \
          \ntrain_loss = {:.4f},     train_f1 = {:.4f},    train_macro_f1 = {:.4f}  \
          \neval_loss = {:.4f},     eval_f1 = {:.4f},    val_f1_macro = {:.4f}  \
              \nlr  =  {:.8f}\nElapsed Time:  {:0>2}:{:0>2}:{:05.2f}"
          .format(epoch, max_epoch, train_loss, train_f1, train_f1_macro, val_loss, val_f1, val_f1_macro, lr, hours,
                  minutes, seconds))


def print_test_stats(test_accuracy, test_precision, test_recall, test_f1, test_f1_macro, best_val_acc,
                     best_val_precision, best_val_recall, best_val_f1):
    print("\nTest accuracy of best model = {:.2f}".format(test_accuracy * 100))
    print("Test precision of best model = {:.2f}".format(test_precision * 100))
    print("Test recall of best model = {:.2f}".format(test_recall * 100))
    print("Test f1 of best model = {:.2f}".format(test_f1 * 100))
    print("Test macro-F1 of best model = {:.2f}".format(test_f1_macro * 100))
    print("\n" + "-" * 50 + "\nBest Validation scores:\n" + "-" * 50)
    print("\nVal accuracy of best model = {:.2f}".format(best_val_acc * 100))
    print("Val precision of best model = {:.2f}".format(best_val_precision * 100))
    print("Val recall of best model = {:.2f}".format(best_val_recall * 100))
    print("Val f1 of best model = {:.2f}".format(best_val_f1 * 100))


def calculate_transformer_stats(train_result):
    train_prec_pos = train_result['tp'] / (train_result['tp'] + train_result['fp'])
    train_recall_pos = train_result['tp'] / (train_result['tp'] + train_result['fn'])
    train_f1_pos = (2 * train_prec_pos * train_recall_pos) / (train_prec_pos + train_recall_pos)
    train_prec_neg = train_result['tn'] / (train_result['tn'] + train_result['fn'])
    train_recall_neg = train_result['tn'] / (train_result['tn'] + train_result['fp'])
    train_f1_neg = (2 * train_prec_neg * train_recall_neg) / (train_prec_neg + train_recall_neg)
    macro_f1 = (train_f1_pos + train_f1_neg) / 2
    return train_prec_pos, train_recall_pos, train_f1_pos, macro_f1


def print_transformer_results(config, val_stats, test_stats, val_result, test_result):
    val_f1_pos, val_f1_neg, val_macro_f1, val_micro_f1, val_recall, val_prec, val_acc = val_stats
    test_f1_pos, test_f1_neg, test_macro_f1, test_micro_f1, test_recall, test_prec, test_acc = test_stats
    val_mcc = val_result['mcc']
    test_mcc = test_result['mcc']

    print("\nVal evaluation stats: \n" + "-" * 50)
    print("Val precision of best model = {:.2f}".format(val_prec * 100))
    print("Val recall of best model = {:.2f}".format(val_recall * 100))
    print("Val f1 (fake) of best model = {:.2f}".format(val_f1_pos * 100))
    print("Val f1 (real) of best model = {:.2f}".format(val_f1_neg * 100))
    print("Val macro-f1 of best model = {:.2f}".format(val_macro_f1 * 100))
    print("Val micro-f1 of best model = {:.2f}".format(val_micro_f1 * 100))
    print("Val accuracy of best model = {:.2f}".format(val_acc * 100))
    print("Val MCC of best model = {:.2f}".format(val_mcc * 100))

    print("\nTest evaluation stats: \n" + "-" * 50)
    print("Test precision of best model = {:.2f}".format(test_prec * 100))
    print("Test recall of best model = {:.2f}".format(test_recall * 100))
    print("Test f1 (fake) of best model = {:.2f}".format(test_f1_pos * 100))
    print("Test f1 (real) of best model = {:.2f}".format(test_f1_neg * 100))
    print("Test macro-f1 of best model = {:.2f}".format(test_macro_f1 * 100))
    print("Test micro-f1 of best model = {:.2f}".format(test_micro_f1 * 100))
    print("Test accuracy of best model = {:.2f}".format(test_acc * 100))
    print("Test MCC of best model = {:.2f}".format(test_mcc * 100))
