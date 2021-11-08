import collections
import pandas as pd
import numpy as np
from tqdm import tqdm
import xlrd
from Bio import motifs


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

def count_ACGT(string):  # Get the percentage of each base in the string groups
    num = {'A': 0, "T": 0, "C": 0, "G": 0}
    for item in string:
        num['A'] = num['A'] + item.upper().count('A')
        num['C'] = num['C'] + item.upper().count('C')
        num['G'] = num['G'] + item.upper().count('G')
        num['T'] = num['T'] + item.upper().count('T')
    total = sum(num.values(), 0.0)
    pro = {k: v / total for k, v in num.items()}
    return pro

def find_locate(seqs, background, file, res_start, res_end, max_read_num, th):
    with open(file) as handle:  # read the pfm matrix
        srf = motifs.read(handle, "pfm")
    pwm = srf.counts.normalize(pseudocounts=0.5)
    pssm = pwm.log_odds(background)

    result = []
    locate = []
    i = 0
    while i < max_read_num:
        try:
            tmp = seqs[res_start:res_end]
        except:
            debug = 0
        tmp = pssm.calculate(tmp)
        result.append(tmp)
        if np.max(tmp) > th:
            locate.append(np.argmax(tmp) + res_start)
        else:
            locate.append(-1)
        i = i + 1
    return result, locate


book1 = xlrd.open_workbook('41592_2018_BFnmeth4633_MOESM3_ESM.xlsx')
sheet = book1.sheet_by_name('Metadata')
nrows = sheet.nrows
tss, oligo = collections.OrderedDict(), collections.OrderedDict()
for i in range(1, nrows):
    seq = str(sheet.cell(i, 20).value.strip())
    id = sheet.cell(i, 0).value
    oligo[seq] = int(id)
book1 = xlrd.open_workbook('41592_2018_BFnmeth4633_MOESM4_ESM.xlsx')
sheet = book1.sheet_by_name('EC')
nrows = sheet.nrows
for i in range(1, nrows):
    id = sheet.cell(i, 0).value
    tss_i = sheet.cell(i, 3).value
    tss[int(id)] = int(tss_i)

f = open('seq_exp_EC.txt')
lines = f.readlines()
seqList = []
seqMask = []
exprList = []
seq_N = ''
bg = ['M']
operators = 'TTGTGAGCGGATAACAA'

stringList = []
for line in tqdm(lines):
    if '>' not in line:
        lt = line.split('\t')
        seq = str(lt[0].strip())
        stringList.append(seq)
background = count_ACGT(stringList)


for line in lines:
    lt = line.split('\t')
    seq = str(lt[0].strip())
    if np.log2(float(lt[1])) > -6 and seq in oligo.keys():
        if tss[oligo[seq]] >= 60:
            tss_i = tss[oligo[seq]]
            results1, locate10 = find_locate(seq, background, '-10.pfm', tss_i - 30, tss_i - 5, 1, -1000)
            results2, locate35 = find_locate(seq, background, '-35.pfm', max(tss_i - 55, 0), tss_i - 30, 1, -1000)
            locate35 = locate35[0]
            locate10 = locate10[0]
            line_mask = seq[:].upper()
            for k in range(0, locate35):
                line_mask = replace_char(line_mask, bg[0], k)
            for k in range(locate35 + 6, locate10):
                line_mask = replace_char(line_mask, bg[0], k)
            for k in range(locate10 + 6, len(line_mask)):
                line_mask = replace_char(line_mask, bg[0], k)
            mask_pos = locate10 + 15
            if mask_pos + len(operators) < len(seq):
                for s in range(len(operators)):
                    line_mask = replace_char(line_mask, seq[:].upper()[mask_pos + s], mask_pos + s)
                seqList.append(seq)
                seqMask.append(line_mask)
                exprList.append(float(lt[1]))
ecoli_data = collections.OrderedDict()
ecoli_data['realA'] = []
ecoli_data['realB'] = []
ecoli_data['expr'] = []
i = 0
bg = ['A', 'T', 'C', 'G']
for line in tqdm(seqList):
    if '>' not in line:
        line_n = line[:].upper()
        if line_n not in ecoli_data['realB']:
            ecoli_data['realB'].append(line_n)
            ecoli_data['realA'].append(seqMask[i])
            ecoli_data['expr'].append(np.log2(exprList[i]))
        i = i + 1
ecoli_data = pd.DataFrame(ecoli_data)
ecoli_data.to_csv('../ecoli_mpra_inducible.csv', index=False)