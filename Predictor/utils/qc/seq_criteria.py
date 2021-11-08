# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 09:53:23 2021

@author: wangpeng884112
"""

import pandas as pd
import numpy as np
import Levenshtein
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FuncFormatter


class Data_predeal:
    def __init__(self, promoter_file):
        self.promoter_file = promoter_file
    
    def output_fa(self, ):
        info = pd.read_table(self.promoter_file,sep =',')
        with open('fakeB.fa','w') as f:
            for i,row in info.iterrows():
                generated = row['fakeB']
                f.write('>' + str(i) + '\n')
                f.write(generated + '\n')
        
        with open('fakeB.sites','w') as f:
            for i,row in info.iterrows():
                generated = row['fakeB']
                f.write('>' + str(i) + '\n')
                f.write(generated + '\n')      
            
        with open('realA.fa','w') as f:
            for i,row in info.iterrows():
                generated = row['realA']
                f.write('>' + str(i) + '\n')
                f.write(generated + '\n')
        
        with open('promoter_100bp.fa','r') as f:
            with open('promoter_100bp.sites','w') as out_f:
                for i,item in enumerate(f):
                    if not '>' in item:
                        item = item.upper()
                        out_f.write('>' + str(i) + '\n')
                        out_f.write(item)
                        

class Edit_distance:
    def read_fa(self,file_name):
        with open(file_name, 'r') as f:
            sequence = []
            for item in f:
                if '>' in item:
                    pass
                else:
                    sequence.append(item.strip('\n'))
            return sequence
                
    def pair_dissim_distribution(self, str_A, str_B):
        i = 0
        start, end = self.find_start_end(str_A)
        str_len = end[0] - start[0]
        dissim_distribution = np.zeros((str_len,))
        for i in range(len(str_A)):
            str_len = end[i] - start[i]
            result = Levenshtein.editops(str_A[i][start[i]:end[i]], str_B[i][start[i]:end[i]]) # catch the editting steps of two sequences
            record = []  # Get the different region
            for item in result:
                if item [0] == 'delete' or item[0] == 'replace' or (item[0] == 'insert' and item[2] <= len(str)):
                    record.append(item[1])
                    record = list(set(record)) #remove repeating location
                    
            tmp_dissim_region = []  # Get the dissimilar region in the str_A
            for k in range(0,str_len):
                if k in record:
                    tmp_dissim_region.append(k)
            for item in tmp_dissim_region:
                dissim_distribution[item] = dissim_distribution[item] + 1
        return dissim_distribution     
    
    # Find the start and end location of the operator
    def find_start_end(self, string):
        start = []
        end = []
        operator = 'ACGT'
        for seq in string:
            flag = 0
            for i,item in enumerate(seq):
                if item in operator and flag == 0:
                    start.append(i)
                    flag = 1
                elif item not in operator and flag == 1:
                    end.append(i)
                    flag = 0
                elif i == len(seq) - 1 and flag == 1:
                    end.append(i + 1)
                    flag = 0
        return start,end
                
            
        
        
        
        
    
matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15) 

font2 = {'family' : 'Arail',
'weight' : 'normal',
'size'   : 20,
}

class Kmer_polyN:  
    def __init__(self):
        pass
    '''
    递归的找出最终的对应数字
    相当于将四进制转换为十进制
    T：3 C:2 G:1 A:0
    左起为高位
    函数输入：
    now_str:已经处理过后的字符串长度
    whole_str:需要处理的全部字符串
    '''
    def find_num(self, now_str,whole_str,num):
        deal_locate = len(whole_str) - len(now_str) - 1
        if whole_str[deal_locate] == 'T' or whole_str[deal_locate] == 't':
            tmp_num = 3
        elif whole_str[deal_locate] == 'A' or whole_str[deal_locate] == 'a':
            tmp_num = 2
        elif whole_str[deal_locate] == 'C' or whole_str[deal_locate] == 'c':
            tmp_num = 1
        else:
            tmp_num = 0
        num = num + tmp_num * np.power(4,len(whole_str) - deal_locate - 1)
        now_str = now_str + whole_str[deal_locate]
        if len(now_str) == len(whole_str):
            return [now_str,whole_str,num]
        else:
            [now_str,whole_str,num] = self.find_num(now_str,whole_str,num)
            return [now_str,whole_str,num]
    
    '''
    函数功能：将二进制代码转换为TCGA，其中T：11；C：10；G：01；A：00
    '''
    def bin_to_di(self, tmp,k_num):
        if len(tmp) % 2 != 0:
            tmp = '0' + tmp
        l = len(tmp)
        result = ''
        while l > 0:
            if tmp[l-2:l] == '11':
                result = 'T' + result
            elif tmp[l-2:l] == '10':
                result = 'A' + result
            elif tmp[l-2:l] == '01':
                result = 'C' + result
            else:
                result = 'G' + result
            l = l - 2
        l = len(tmp)
        remain = k_num - l/2
        result = int(remain) * 'G' + result
        return result
    
    def count_kmer(self, promoter_seq, k_num):
        num_kmer = np.zeros((np.power(4,k_num),1))
        iter_promoter = 0
        for iter_promoter in range(len(promoter_seq)):       #统计每一种kmer的个数
            iter_len = 0
            while iter_len < len(promoter_seq[iter_len]) - k_num + 1:
                tmp = promoter_seq[iter_promoter][iter_len:iter_len + k_num]
                now_str = ''
                num = 0
                now_str,whole_str,num = self.find_num(now_str,tmp,num)
                num_kmer[num] = num_kmer[num] + 1
                iter_len = iter_len + 1
     
        i = 0
        name_kmer = []     #找出每一个数字对应的kmer名称
        while i < np.power(4, k_num):
            tmp = bin(i)[2::]
            tmp = self.bin_to_di(tmp,k_num)
            name_kmer.append(tmp)
            i = i + 1
        
        num_kmer = num_kmer[:,0]        #找出出现次数最多的kmer
        sort = np.argsort(num_kmer)
        sort = sort[::-1]
        name_kmer_sort = []
        i = 0
        while i < len(name_kmer):
            name_kmer_sort.append(name_kmer[sort[i]])
            i = i + 1
        
        return num_kmer,name_kmer,name_kmer_sort

    def draw_picture(self, num_kmer, label, select_color = 'blue'):
        num_kmer = num_kmer / max(num_kmer)
        x = range(len(num_kmer))
        y = num_kmer/sum(num_kmer)
        plt.figure(1)
    #    figure, ax = plt.plots()
        plt.xlabel('kmer index',font2)
        plt.ylabel('kmer frequency',font2)
        plt.plot(x,y,color=select_color,alpha=0.7,label=label)
        plt.legend()


from Bio import motifs
from Bio.Seq import Seq
from scipy.stats import gaussian_kde


class space_distance:
    def __init__(self, string, sequence_file,color,label):
        self.max_read_num = 5000
        self.res_start_35 = 35
        self.res_end_35 = 50
        self.res_start_10 = 60
        self.res_end_10 = 70
        self.th = -1000
        self.background = self.count_ACGT(string)
        self.sequence_file = sequence_file
        self.color = color
        self.label = label
        
    def count_ACGT(self,string):  #Get the percentage of each base in the string groups
        num = {'A':0,"T":0,"C":0,"G":0}
        for item in string:
            num['A'] = num['A'] + item.upper().count('A')
            num['C'] = num['C'] + item.upper().count('C')
            num['G'] = num['G'] + item.upper().count('G')
            num['T'] = num['T'] + item.upper().count('T')
        total = sum(num.values(), 0.0)
        pro = {k: v / total for k, v in num.items()}
        return pro
    
    # 控制参数输入
    def input_para(self):
        with open(self.sequence_file) as handle:
            arnt = motifs.read(handle, "sites")    
        # 计算-10区域和-35区域在每个promoter中的位置
        file = '-35.pfm'
        result_35,locate_35 = self.find_locate(arnt,file,self.res_start_35,self.res_end_35,self.max_read_num,self.th)   
        file = '-10.pfm'
        result_10,locate_10 = self.find_locate(arnt,file,self.res_start_10,self.res_end_10,self.max_read_num,self.th)      
        # 画差值位置分布图
        distance = self.draw_dif_distribution(locate_10,locate_35,100)
        return locate_10,locate_35,distance
    
    # 找到motif位置
    def find_locate(self,arnt,file,res_start,res_end,max_read_num,th):
        with open(file) as handle:   #read the pfm matrix
             srf = motifs.read(handle, "pfm")        
        pwm = srf.counts.normalize(pseudocounts=0.5)    
        pssm = pwm.log_odds(self.background)
        print(pssm)
            
        result = []
        locate = []
        i = 0
        while i < max_read_num:
            try:
                tmp = arnt.instances[i][res_start:res_end]
            except:
                print(i)
                print(arnt.instances[i])
                print(res_start)
                print(res_end)
            tmp = pssm.calculate(tmp)
            result.append(tmp)
            if np.max(tmp) > th:
                locate.append(np.argmax(tmp) + res_start)
            else:
                locate.append(-1)
            i = i + 1
        return result,locate

    # 画图函数
    def draw_dif_distribution(self,locate_10,locate_35,p):
        distance = []
        i = 0
        while i < len(locate_10):
            if locate_10[i] != -1 and locate_35[i] != -1:
                distance.append(locate_10[i] - locate_35[i] - 6)
            i = i + 1
        distance = np.matrix(distance)
        distance = distance.T
        
        #绘制density plot
        distance = distance.T
        distance = np.array(distance)
        distance = distance.T
        distance = distance[:,0]
        
        density = gaussian_kde(distance)
        xs = np.linspace(0,30,200)
        density.covariance_factor = lambda : 0.5
        density._compute_covariance()
        
        plt.figure(p)
        plt.xlabel('kmer index',font2)
        plt.ylabel('kmer frequency',font2)
        plt.plot(xs,density(xs),color=self.color,label=self.label)
        plt.legend()
        return density


class poly_stat:
    def __init__(self, sequence):
        self.sequence = sequence
        self.sample_num = len(sequence)
        self.seq_len = len(sequence[0])
        


    def poly_statistic(self, seq, polyN):
        for n in range(self.sample_num):
            for item in polyN:
                item = item.upper()
                for i in range(self.seq_len - len(item) + 1):
                    kmr = seq[n][i:i + len(item)]
                    if kmr == item:
                        polyN[item] += 1  
    
    def poly_N(self, info_label,mode):
        ## poly-T statistics
        if mode == 'polyT':
            polyN = {'TTTTT':0,'TTTTTT':0,'TTTTTTT':0,'TTTTTTTT':0}
        else:
        ## poly-A statistics
            polyN = {'AAAAA':0,'AAAAAA':0,'AAAAAAA':0,'AAAAAAAA':0}
        self.poly_statistic(self.sequence, polyN)
        lists = sorted(polyN.items())
        x, y = zip(*lists)
        plt.plot(x, (1/self.sample_num) * np.asarray(y), label = info_label)
        
        
    
def seq_criteria(promoter_file):
    predeal = Data_predeal(promoter_file)
    predeal.output_fa()
    
    ed = Edit_distance()
    real_control = ed.read_fa('promoter_100bp.fa')
    realA = ed.read_fa('realA.fa')
    fakeB = ed.read_fa('fakeB.fa')

    # k-mer分布，与天然序列做的比对
    kmer = Kmer_polyN()
    num_kmer_real,name_kmer,name_kmer_sort = kmer.count_kmer(real_control, 4)
    num_kmer_fake,name_kmer,name_kmer_sort = kmer.count_kmer(fakeB, 4)
    kmer.draw_picture(num_kmer_fake, label = 'fake', select_color='red')
    kmer.draw_picture(num_kmer_real, label = 'real', select_color='blue')
    plt.title('Kmer Distribution') 
    
    
    # 查看realA给出条件部分，与fakeB的条件部分的差异区域分布
    pair_dissim_distribution = ed.pair_dissim_distribution(realA,fakeB)
    x = list(range(17))
    plt.figure(2)
    plt.plot(x,pair_dissim_distribution)
    plt.ylim([0,100])
    plt.xlabel('Location')
    plt.ylabel('dissimilarity counts')
    plt.title('Dissimilarity Region') 
    
    # -10，-35区域距离
    sp = space_distance(real_control, 'promoter_100bp.sites','blue','real')
    locate_10,locate_35,distance = sp.input_para()
    sp = space_distance(fakeB, 'fakeB.sites','red','fake')
    locate_10,locate_35,distance = sp.input_para()
    plt.title('-10 -35 distance distribtuion') 
    
    #poly-T and poly-A
    plt.figure(3)
    case = poly_stat(fakeB)
    control = poly_stat(real_control)
    case.poly_N('case','polyT')
    control.poly_N('control','polyT')
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left",prop = {'size':12})
    plt.title('poly-T') 
    plt.show()

    plt.figure(4)
    case.poly_N('case','polyA')
    control.poly_N('control','polyA')
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left",prop = {'size':12})
    plt.title('poly-A')
    plt.show()


    