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
    data_path = '../ecoli_mpra_inducible/ecoli_mpra_prediction_fold_{}.csv'
    k_fold = 3
    evaluate_number = 1
    operator = 'TTGTGAGCGGATAACAA'
    datasets = []
    for i in range(k_fold):
        datasets.append(pd.read_csv(data_path.format(i), index_col=0))
    for fold_i in range(k_fold):
        is_evaluate_data_fold_i = False
        for i in range(k_fold):
            if i != fold_i:
                if not is_evaluate_data_fold_i:
                    evaluate_data_fold_i = deepcopy(datasets[i])
                    is_evaluate_data_fold_i = True
                else:
                    evaluate_data_fold_i = evaluate_data_fold_i.append(datasets[i])
        index = list(evaluate_data_fold_i.index)
        random.shuffle(index)
        index = index[0 : evaluate_number]
        realA = list(evaluate_data_fold_i.loc[index]['realA'])
        realB = list(evaluate_data_fold_i.loc[index]['realB'])
        new_realA, new_realB = [], []
        for i in range(len(realA)):
            A_i, B_i = realA[i], realB[i]
            for j in range(len(realA[0])):
                if A_i[len(realA[0]) - j - 1] != 'M':
                    break
            startIdx = len(realA[0]) - j - 1 - (len(operator) - 1)
            for s in range(len(operator)):
                A_i = replace_char(A_i, operator[s], startIdx + s)
                B_i = replace_char(B_i, operator[s], startIdx + s)
            new_realA.append(A_i)
            new_realB.append(B_i)
        new_evaluate_fold_i = collections.OrderedDict()
        new_evaluate_fold_i['realA'] = new_realA
        new_evaluate_fold_i['realB'] = new_realB
        new_evaluate_fold_i = pd.DataFrame(new_evaluate_fold_i)
        new_evaluate_fold_i.to_csv('../ecoli_mpra_inducible/ecoli_mpra_prediction_test_fold_{}.csv'.format(fold_i))



if __name__ == '__main__':
    main()