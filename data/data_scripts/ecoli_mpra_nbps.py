import pandas as pd
import collections
import numpy as np


def replace_char(string, char, index):
    string = np.asarray(list(string))
    string[index] = char
    return ''.join(list(string))


def main():
    path = 'ecoli_mpra.csv'
    cache_dir = '../ecoli_mpra_nbps/ecoli_mpra_nbps.csv'
    polish_expectation = 10
    prob_num = 5
    sample_num = 5
    natural_seq = list(pd.read_csv(path)['realB'])
    seq_L = len(natural_seq[0])
    broken_seq = collections.OrderedDict()
    broken_seq['realA'], broken_seq['realB'] = [], []
    maxPolishE = min(polish_expectation, seq_L)
    for seq_i in natural_seq:
        for j in range(prob_num):
            prob_i = np.random.rand(seq_L)
            prob_i = maxPolishE / np.sum(prob_i) * prob_i
            prob_i[prob_i > 1] = 1
            for k in range(sample_num):
                n_trial = np.ones(len(seq_i))
                broken_mask = np.random.binomial(list(n_trial), list(prob_i))
                idx = np.arange(seq_L)
                idx = idx[broken_mask == 1]
                seq_i_broken = replace_char(seq_i, 'M', idx)
                broken_seq['realA'].append(seq_i_broken)
                broken_seq['realB'].append(seq_i)
    broken_seq = pd.DataFrame(broken_seq)
    broken_seq.to_csv(cache_dir.format(polish_expectation))


if __name__ == '__main__':
    main()