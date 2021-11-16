import polisher_module
from SeqRegressionModel import *
from wgan_attn import *
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import ConcatDataset
import torch
import random
from torch.utils.data import DataLoader
import collections
import pandas as pd
from sko.GA import GA
import pickle


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def main():
    k_fold = 3
    train_fold_path = "../data/ecoli_mpra_nbps/ecoli_mpra_nbps_prediction_train_fold_{}.csv"
    val_fold_path = '../data/ecoli_mpra_nbps/ecoli_mpra_nbps_prediction_val_fold_{}.csv'
    opt_predictor_path = '../Metrics/results/model/ecoli_mpra_nbps_prediction_opt_{}.pth'
    metric_predictor_path = '../Metrics/results/model/ecoli_mpra_nbps_prediction_metric_{}.pth'
    generator_path = "../Generator/check_points/ecoli_mpra_nbps/net_G_7399.pth"
    eval_number = 5
    results_save_path = 'results/ecoli_mpra_nbps_evaluation_fold_{}.csv'
    mode = 'search_nbps'
    polishE = 10

    for i in range(k_fold):
        evaluate_data_fold_i = pd.read_csv(val_fold_path.format(i))
        idx = list(evaluate_data_fold_i.index)
        random.shuffle(idx)
        if mode == 'search_nbps':
            polish_seq, control_seq = list(evaluate_data_fold_i['realA'])[0: eval_number], list(evaluate_data_fold_i['realA'])[0: eval_number]
            op = polisher_module.optimizer_search_nbps(predictor_path=opt_predictor_path.format(i),
                                       generator_path=generator_path,
                                       size_pop=500,
                                       max_iter=50,
                                       polishE=polishE)
        elif mode == 'fix_blank':
            polish_seq, control_seq = list(evaluate_data_fold_i['realA_OP'])[0: eval_number], list(evaluate_data_fold_i['realB_OP'])[0: eval_number]
            op = polisher_module.optimizer_fix_blank(predictor_path=opt_predictor_path.format(i),
                                           generator_path=generator_path,
                                           size_pop=500,
                                           max_iter=50)
        op.set_input(polish_seq, control_seq)
        op.optimization()
        results = collections.OrderedDict()
        results['control'], results['case'] = [], []
        predictor = torch.load(metric_predictor_path.format(i))

        with torch.no_grad():
            for seq in polish_seq:
                control_seq = op.control_results[seq]
                control_seq = transforms.ToTensor()(polisher_module.one_hot(control_seq)).float().cuda()
                case_seq = op.seq_results[seq][0]
                case_seq = transforms.ToTensor()(polisher_module.one_hot(case_seq)).float().cuda()

                control_exp = predictor(control_seq)
                case_exp = predictor(case_seq)
                results['control'].append(control_exp.cpu().float().numpy()[0])
                results['case'].append(case_exp.cpu().float().numpy()[0])
            results = pd.DataFrame(results)
            results.to_csv(results_save_path.format(i))


if __name__ == '__main__':
    main()