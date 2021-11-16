import pandas as pd
import logging
import time
import collections
from matplotlib import pyplot as plt
import numpy as np
import torch


DICT_DEFAULT = {'TATAAT': 'TTGACA', 'AATTTA': 'TTACCA', 'GCGATA': 'CAATGG', 'TGGAAT': 'TGTGTA',
                'TCATCT': 'AAACGC', 'ACCTGG': 'GTTCCC', 'TCGGAT': 'TATCGA'}


def relation_reserve(csv_path, st=[46], ed=[52]):
    path = csv_path
    results = pd.read_csv(path)
    fakeB = list(results['fakeB'])
    realA = list(results['realA'])
    realB = list(results['realB'])
    n, cn = 0, 0
    for i in range(len(realB)):
        realAt, realBt, fakeBt = realA[i], realB[i], fakeB[i]
        for j in range(len(st)):
            for k in range(st[j], ed[j], 1):
                n = n + 1
                if fakeBt[k] == realBt[k]:
                    cn = cn + 1
    return cn/n


def polyAT_freq(valid_path, ref_path):
    A_dict_valid = {'AAAAA':0, 'AAAAAA':0, 'AAAAAAA': 0, 'AAAAAAAA': 0}
    A_dict_ref = {'AAAAA': 0, 'AAAAAA': 0, 'AAAAAAA': 0, 'AAAAAAAA': 0}
    T_dict_valid = {'TTTTT':0, 'TTTTTT':0, 'TTTTTTT': 0, 'TTTTTTTT': 0}
    T_dict_ref = {'TTTTT': 0, 'TTTTTT': 0, 'TTTTTTT': 0, 'TTTTTTTT': 0}
    valid_df = pd.read_csv(valid_path)
    ref_df = pd.read_csv(ref_path)
    fakeB = list(valid_df['fakeB'])
    realB = list(ref_df['realB'])
    for i in range(len(fakeB)):
        fakeBt = fakeB[i]
        for keys in A_dict_valid.keys():
            for j in range(0, len(fakeBt) - len(keys) + 1):
                if fakeBt[j : j + len(keys)] == keys:
                    A_dict_valid[keys] += 1
        for keys in T_dict_valid.keys():
            for j in range(0, len(fakeBt) - len(keys) + 1):
                if fakeBt[j : j + len(keys)] == keys:
                    T_dict_valid[keys] += 1
    for i in range(len(realB)):
        realBt = realB[i]
        for keys in A_dict_ref.keys():
            for j in range(0, len(realBt) - len(keys) + 1):
                if realBt[j : j + len(keys)] == keys:
                    A_dict_ref[keys] += 1
        for keys in T_dict_ref.keys():
            for j in range(0, len(realBt) - len(keys) + 1):
                if realBt[j : j + len(keys)] == keys:
                    T_dict_ref[keys] += 1

    for keys in A_dict_valid.keys():
        A_dict_valid[keys] = A_dict_valid[keys] / len(fakeB)
        A_dict_ref[keys] = A_dict_ref[keys] / len(realB)
    for keys in T_dict_valid.keys():
        T_dict_valid[keys] = T_dict_valid[keys] / len(fakeB)
        T_dict_ref[keys] = T_dict_ref[keys]/len(realB)

    return A_dict_valid, A_dict_ref, T_dict_valid, T_dict_ref


def get_logger(log_name='.log'):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def convert(n, x):
    list_a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'A', 'b', 'C', 'D', 'E', 'F']
    list_b = []
    while True:
        s, y = divmod(n, x)
        list_b.append(y)
        if s == 0:
            break
        n = s
    list_b.reverse()
    res = []
    for i in range(x):
        res.append(0)
    res0 = []
    for i in list_b:
        res0.append(list_a[i])
    for i in range(len(res0)):
        res[x - i - 1] = res0[len(res0) - i - 1]
    return res


def kmer_frequency(valid_path, ref_path, k=4, save_path='cache/', save_name='99'):
    print('Start saving the frequency figure......')
    bg = ['A', 'T', 'C', 'G']
    valid_kmer, ref_kmer = collections.OrderedDict(), collections.OrderedDict()
    kmer_name = []
    for i in range(4**k):
        nameJ = ''
        cov = convert(i, 4)
        for j in range(k):
                nameJ += bg[cov[j]]
        kmer_name.append(nameJ)
        valid_kmer[nameJ], ref_kmer[nameJ] = 0, 0
    valid_df = pd.read_csv(valid_path)
    ref_df = pd.read_csv(ref_path)
    fakeB = list(valid_df['fakeB'])
    realB = list(ref_df['realB'])
    realA = list(valid_df['realA'])
    valid_num, ref_num = 0, 0
    for i in range(len(fakeB)):
        for j in range(len(fakeB[0]) - k + 1):
            k_mer = fakeB[i][j : j + k]
            mask_A = realA[i][j : j + k]
            if 'A' not in mask_A and 'T' not in mask_A and 'C' not in mask_A and 'G' not in mask_A:
                valid_kmer[k_mer] += 1
                valid_num += 1
    for i in range(len(realB)):
        for j in range(len(realB[0]) - k + 1):
            k_mer = realB[i][j : j + k]
            ref_num += 1
            ref_kmer[k_mer] += 1
    for i in kmer_name:
        ref_kmer[i], valid_kmer[i] = ref_kmer[i]/ref_num, valid_kmer[i]/valid_num
    plt.plot(list(ref_kmer.values()))
    plt.plot(list(valid_kmer.values()))
    plt.legend(['real distribution', 'model distribution'])
    plt.title('{}_mer frequency'.format(k))
    plt.xlabel('{}_mer index'.format(k))
    plt.ylabel('{}_mer frequency'.format(k))
    plt.savefig('{}{}_{}_mer_frequency.png'.format(save_path, save_name, k))
    plt.close()


def tensor2seq(input_sequence, label):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_sequence, np.ndarray):
        if isinstance(input_sequence, torch.Tensor):  # get the data from a variable
            sequence_tensor = input_sequence.data
        else:
            return input_sequence
        sequence_numpy = sequence_tensor.cpu().float().numpy()  # convert it into a numpy array
    else:  # if it is a numpy array, do nothing
        sequence_numpy = input_sequence
    return decode_oneHot(sequence_numpy, label)


def reserve_percentage(tensorInput, tensorSeq, blankSym='M'):
    results =collections.OrderedDict()
    results['fakeB'] = []
    results['realA'] = []
    for seqT in tensorSeq:
        label = 'fakeB'
        for j in range(seqT.size(0)):
            seq = tensor2seq(torch.squeeze(seqT[j, :, :]), label)
            results[label].append(seq)
    for seqT in tensorInput:
        label = 'realA'
        for j in range(seqT.size(0)):
            seq = tensor2seq(torch.squeeze(seqT[j, :, :]), label)
            results[label].append(seq)
    c, n = 0.0, 0.0
    for i in range(len(results['fakeB'])):
        seqA = results['realA'][i]
        seqB = results['fakeB'][i]
        for j in range(len(seqA)):
            if seqA[j] != blankSym:
                n += 1
                if seqA[j] == seqB[j]:
                    c += 1
    return 100*c/n


def save_sequence(tensorSeq, tensorInput, tensorRealB, save_name='', cut_r=0.1):
    i = 0
    results =collections.OrderedDict()
    results['fakeB'] = []
    results['realA'] = []
    results['realB'] = []
    for seqT in tensorSeq:
        label = 'fakeB'
        for j in range(seqT.size(0)):
            seq = tensor2seq(torch.squeeze(seqT[j, :, :]), label)
            results[label].append(seq)
        i = i + 1
    for seqT in tensorInput:
        label = 'realA'
        for j in range(seqT.size(0)):
            seq = tensor2seq(torch.squeeze(seqT[j, :, :]), label)
            results[label].append(seq)
        i = i + 1
    for seqT in tensorRealB:
        label = 'realB'
        for j in range(seqT.size(0)):
            seq = tensor2seq(torch.squeeze(seqT[j, :, :]), label)
            results[label].append(seq)
        i = i + 1
    for label in ['realA', 'fakeB', 'realB']:
        results[label] = results[label][0 : int(cut_r * len(results[label]))]
    results = pd.DataFrame(results)
    results.to_csv(save_name, index=False)
    return save_name


def decode_oneHot(seq, label):
    keys, dSeq = ['A', 'T', 'C', 'G'], ''
    for i in range(np.size(seq, 1)):
        if label == 'realA':
            if np.max(seq[:, i]) != 1:
                dSeq += 'M'
            else:
                dSeq += keys[np.argmax(seq[:, i])]
        else:
            dSeq += keys[np.argmax(seq[:, i])]
    return dSeq


def csv2fasta(csv_path, data_path, data_name):
    path = csv_path
    results = pd.read_csv(path)
    fakeB = list(results['fakeB'])
    realB = list(results['realB'])
    f2 = open(data_path + data_name + '_fakeB.fasta','w')
    j = 0
    for i in fakeB:
        f2.write('>sequence_generate_'+str(j) + '\n')
        f2.write(i + '\n')
        j = j + 1
    f2.close()
    f2 = open(data_path + data_name + '_realB.fasta', 'w')
    j = 0
    for i in realB:
        f2.write('>sequence_generate_' + str(j) + '\n')
        f2.write(i + '\n')
        j = j + 1
    f2.close()


def log_polyAT(logger, A_dict_valid, A_dict_ref, T_dict_valid, T_dict_ref):
    logger.info('polyA valid AAAAA:{} AAAAAA:{} AAAAAAA:{} AAAAAAAA:{}'.format(A_dict_valid['AAAAA'],
                                                                               A_dict_valid['AAAAAA'],
                                                                               A_dict_valid['AAAAAAA'],
                                                                               A_dict_valid['AAAAAAAA']))
    logger.info('polyA ref AAAAA:{} AAAAAA:{} AAAAAAA:{} AAAAAAAA:{}'.format(A_dict_ref['AAAAA'],
                                                                             A_dict_ref['AAAAAA'],
                                                                             A_dict_ref['AAAAAAA'],
                                                                             A_dict_ref['AAAAAAAA']))
    logger.info('polyT valid TTTTT:{} TTTTTT:{} TTTTTTT:{} TTTTTTTT:{}'.format(T_dict_valid['TTTTT'],
                                                                               T_dict_valid['TTTTTT'],
                                                                               T_dict_valid['TTTTTTT'],
                                                                               T_dict_valid['TTTTTTTT']))
    logger.info('polyT ref TTTTT:{} TTTTTT:{} TTTTTTT:{} TTTTTTTT:{}'.format(T_dict_ref['TTTTT'],
                                                                             T_dict_ref['TTTTTT'],
                                                                             T_dict_ref['TTTTTTT'],
                                                                             T_dict_ref['TTTTTTTT']))
