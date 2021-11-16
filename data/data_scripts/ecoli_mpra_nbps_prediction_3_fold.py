import pandas as pd
import random
from copy import deepcopy
import collections


def replace_char(string, char, index):
    string = list(string)
    string[index] = char
    return ''.join(string)


def replace_string(string, sub_string, index):
    string = list(string)
    for i in range(len(sub_string)):
        string[index + i] = sub_string[i]
    return ''.join(string)


def multi_sub(string, p_start, p_end,  c):
    new = []
    for s in string:
        new.append(s)
    for i in range(p_start, p_end + 1, 1):
        new[i] = c
    return ''.join(new)


def main():
    data_path = 'ecoli_mpra.csv'
    k_fold = 3
    datasets = pd.read_csv(data_path)
    index = list(datasets.index)
    total_number = len(index)
    fold_size = int(1 / k_fold * total_number)
    index_0 = deepcopy(index)
    random.shuffle(index_0)
    for fold_i in range(k_fold):
        data_val= collections.OrderedDict()
        index_val_i = index_0[fold_i*fold_size: (fold_i + 1)*fold_size]
        expr = list(datasets.loc[index_val_i]['expr'])
        seqs = list(datasets.loc[index_val_i]['realB'])
        data_val['realB'] = seqs
        data_val['expr'] = expr
        data_val['realA'] = list(datasets.loc[index_val_i]['realB'])
        data_val = pd.DataFrame(data_val)


        index_train_i = []
        for train_i in range(k_fold):
            if train_i != fold_i:
                index_train_i += index_0[train_i*fold_size: (train_i + 1)*fold_size]
        data_train = collections.OrderedDict()
        expr = list(datasets.loc[index_train_i]['expr'])
        seqs = list(datasets.loc[index_train_i]['realB'])
        data_train['realB'] = seqs
        data_train['expr'] = expr
        data_train['realA'] = list(datasets.loc[index_train_i]['realB'])
        data_train = pd.DataFrame(data_train)

        data_train.to_csv('../ecoli_mpra_nbps/ecoli_mpra_nbps_prediction_train_fold_{}.csv'.format(fold_i), index=False)
        data_val.to_csv('../ecoli_mpra_nbps/ecoli_mpra_nbps_prediction_val_fold_{}.csv'.format(fold_i), index=False)



if __name__ == '__main__':
    main()