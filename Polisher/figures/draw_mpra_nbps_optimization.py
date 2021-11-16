from matplotlib import pyplot as plt
import numpy as np
import collections
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
import pandas as pd


#mpl.rcParams['pdf.fonttype'] = 42
print(mpl.matplotlib_fname())
print(mpl.get_cachedir())
MEDIUM_SIZE = 8
SMALLER_SIZE = 6
BIGGER_SIZE = 25
plt.rc('font', family='Helvetica', size=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)	# fontsize of the axes title
plt.rc('xtick', labelsize=SMALLER_SIZE)	# fontsize of the tick labels
plt.rc('ytick', labelsize=SMALLER_SIZE)	# fontsize of the tick labels
plt.rc('figure', titlesize=MEDIUM_SIZE)
plt.rc('legend', fontsize=MEDIUM_SIZE)
FIG_HEIGHT = 2
FIG_WIDTH = 2




def main():
    results_path = '../results/ecoli_mpra_nbps_evaluation_fold_{}.csv'

    k_fold = 3
    for fold_i in range(k_fold):

        results = pd.read_csv(results_path.format(fold_i), index_col=0)
        mean = np.asarray(results)

        plt.figure(figsize=[6, 6])


        fig, ax = plt.subplots(figsize=(FIG_WIDTH * 1.2, FIG_HEIGHT))

        nmethod, ngroup = np.shape(mean)

        n_groups = 5
        nmethod = 2
        index = np.arange(n_groups)
        bar_width = 1. / nmethod * 0.8
        opacity = 0.8

        ax.bar(index + (nmethod - 1 - 2) * bar_width, mean[:, 0], width=bar_width, alpha=opacity,
               color='#018571',  # ,color_l[i],
               label='Origin')
        ax.bar(index + (nmethod - 1 - 1) * bar_width, mean[:, 1], width=bar_width, alpha=opacity,
               color='#a6611a',  # ,color_l[i],
               label='Our design')
        csfont = {'family': 'Helvetica'}


        if nmethod == 1:
            ax.set_xticks(index)
        else:
            ax.set_xticks(index + bar_width * (nmethod - 0.5) * 1. / 2 - 0.1)
        plt.legend(loc='upper right', frameon=False, ncol=1, fontsize=4)
        # plt.legend(loc='upper left',bbox_to_anchor=(0.1, 1.1), frameon=False, ncol=1, fontsize=4)

        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", va="center",
                 rotation_mode="anchor")

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # ax.legend(method_l)

        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
        #title_set = {'data-cafa_cat': 'Data-Cafa', 'data_2016_cat': 'data_2016', 'goa_human_cat': 'goa human'}
        title_set = 'Fold: {}'.format(fold_i)
        ax.set_title(title_set, fontdict=csfont)
        ax.set_ylabel('Expredssion (Log)', fontdict=csfont)
        ax.set_xticklabels(['1', '2', '3', '4', '5'])
        ax.set_xlabel('promoter', fontdict=csfont)
        #ax.xaxis.set_visible(False)
        #ax.yaxis.set_visible(False)

        fig.tight_layout()
        #plt.xlabel('AUROC Greater Than')
        #plt.ylabel('AUROC Percentage')
        #plt.legend(['SDN', 'deepgoplus+Network', 'deepgoplus+TFIDF'])
        #plt.title('Data-Cafa')
        #ax.xaxis.set_visible(False)
        #ax.yaxis.set_visible(False)
        #plt.show()
        plt.savefig('plot_results/mpra_nbps_fold_{}.pdf'.format(fold_i))


if __name__ == '__main__':
        main()