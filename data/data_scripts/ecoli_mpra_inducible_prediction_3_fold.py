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
    data_path = '../ecoli_mpra_inducible/ecoli_mpra_inducible.csv'
    operator = 'TTGTGAGCGGATAACAA'
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
        data_val['realA'] = list(datasets.loc[index_val_i]['realA'])
        data_val = pd.DataFrame(data_val)
        new_realA, new_realB = [], []
        for i in range(len(data_val['realA'])):
            A_i, B_i = data_val['realA'][i], data_val['realB'][i]
            for j in range(len(data_val['realA'][0])):
                if A_i[len(data_val['realA'][0]) - j - 1] != 'M':
                    break
            startIdx = len(data_val['realA'][0]) - j - 1 - (len(operator) - 1)
            for s in range(len(operator)):
                A_i = replace_char(A_i, operator[s], startIdx + s)
                B_i = replace_char(B_i, operator[s], startIdx + s)
            new_realA.append(A_i)
            new_realB.append(B_i)
        data_val['realA_OP'] = new_realA
        data_val['realB_OP'] = new_realB


        index_train_i = []
        for train_i in range(k_fold):
            if train_i != fold_i:
                index_train_i += index_0[train_i*fold_size: (train_i + 1)*fold_size]
        data_train = collections.OrderedDict()
        expr = list(datasets.loc[index_train_i]['expr'])
        seqs = list(datasets.loc[index_train_i]['realB'])
        data_train['realB'] = seqs
        data_train['expr'] = expr
        data_train['realA'] = list(datasets.loc[index_train_i]['realA'])
        data_train = pd.DataFrame(data_train)
        new_realA, new_realB = [], []
        for i in range(len(data_train['realA'])):
            A_i, B_i = data_train['realA'][i], data_train['realB'][i]
            for j in range(len(data_train['realA'][0])):
                if A_i[len(data_train['realA'][0]) - j - 1] != 'M':
                    break
            startIdx = len(data_train['realA'][0]) - j - 1 - (len(operator) - 1)
            for s in range(len(operator)):
                A_i = replace_char(A_i, operator[s], startIdx + s)
                B_i = replace_char(B_i, operator[s], startIdx + s)
            new_realA.append(A_i)
            new_realB.append(B_i)
        data_train['realA_OP'] = new_realA
        data_train['realB_OP'] = new_realB

        data_train.to_csv('../ecoli_mpra_inducible/ecoli_mpra_prediction_train_fold_{}.csv'.format(fold_i), index=False)
        data_val.to_csv('../ecoli_mpra_inducible/ecoli_mpra_prediction_val_fold_{}.csv'.format(fold_i), index=False)



if __name__ == '__main__':
    main()