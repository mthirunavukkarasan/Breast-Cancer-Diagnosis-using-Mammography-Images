import numpy as np
import warnings
import matplotlib
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from prettytable import PrettyTable


def Statistical(val):
    out = np.zeros((5))
    out[0] = max(val)
    out[1] = min(val)
    out[2] = np.mean(val)
    out[3] = np.median(val)
    out[4] = np.std(val)
    return out

def Plot_Image_Results():
    matplotlib.use('TkAgg')
    for a in range(2):
        eval = np.load('Eval_seg.npy', allow_pickle=True)[a]
        Terms = ['Accuracy', 'Dice', 'Jaccard']
        value = eval[ :, :, :]
        stat = np.zeros((value.shape[1], value.shape[2], 5))
        for j in range(value.shape[1]): # For all algms and Mtds
            for k in range(value.shape[2]): # For all terms
                stat[j, k, :] = Statistical(value[:, j, k])
        stat = stat
        learnper = [1,2,3,4,5]
        for k in range(len(Terms)):

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            X = np.arange(5)
            ax.bar(X + 0.00, stat[0, k, :], color='#f97306', width=0.10, label="Unet")
            ax.bar(X + 0.10, stat[1, k, :], color='#cc3f81', width=0.10, label="Unet3+")
            ax.bar(X + 0.20, stat[2, k, :], color='#ccbc3f', width=0.10, label="TransUnet")
            ax.bar(X + 0.30, stat[3, k, :], color='c', width=0.10, label="Trans-Unet++ ")
            ax.bar(X + 0.40, stat[4, k, :], color='k', width=0.10, label="UNet3+-LSA")
            plt.xticks(X + 0.10, ('Best', 'Worst', 'Mean', 'Median', 'Std'))
            plt.ylabel(Terms[k])
            plt.xlabel('Statistical Analysis')
            plt.legend(loc=1)
            path1 = "./Results/Segmented_%s_Image_bar_%s.png" % (str(a+1),Terms[k])
            plt.savefig(path1)
            plt.show()

def plot_results_Batch():
    for a in range(2):
        eval1 = np.load('Eval_all.npy', allow_pickle=True)[a]
        Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
        Graph_Terms = [0,1,2,3,4,5,6,7,8,9]
        Algorithm = ['TERMS', 'RSA-AViT-SNetv2', 'TFMOA-AViT-SNetv2', 'SCO-AViT-SNetv2', 'GOA-AViT-SNetv2', 'MRFGO-AViT-SNetv2']
        Classifier = ['TERMS', 'CNN', 'DENSENET', 'RAN', 'ViT_SNetv2','MRFGO-AViT-SNetv2']

        value1 = eval1[ 4, :, 4:]
        value1[:, :-1] = value1[:, :-1] * 100

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Algorithm[j + 1], value1[j, :])
        print('--------------------------------------------------Dataset_'+str(a+1)+'_Algorithm Comparison',
              '--------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value1[len(Algorithm) + j - 1, :])
        print('--------------------------------------------------Dataset_'+str(a+1)+'_Classifier Comparison',
              '--------------------------------------------------')
        print(Table)
        eval1 = np.load('Eval_all.npy', allow_pickle=True)[a]
        learnper = [1, 2, 3, 4, 5]
        # for i in range(eval1.shape[0]):  # eval1.shape[0]
        for j in range(len(Graph_Terms)):
            Graph = np.zeros((eval1.shape[0], eval1.shape[1]))
            for k in range(eval1.shape[0]):
                for l in range(eval1.shape[1]):
                    if j == 9:
                        Graph[k, l] = eval1[k, l, Graph_Terms[j] + 4]
                    else:
                        Graph[k, l] = eval1[ k, l, Graph_Terms[j] + 4] * 100

            plt.plot(learnper, Graph[:, 0],  '-.',color='#65fe08', linewidth=3, marker='o', markerfacecolor='blue', markersize=12,
                     label="RSA-AViT-SNetv2 ")
            plt.plot(learnper, Graph[:, 1], '-.', color='#4e0550', linewidth=3, marker='o', markerfacecolor='red', markersize=12,
                     label="TFMOA-AViT-SNetv2 ")
            plt.plot(learnper, Graph[:, 2], '-.', color='#f70ffa', linewidth=3, marker='o', markerfacecolor='green', markersize=12,
                     label="SCO-AViT-SNetv2 ")
            plt.plot(learnper, Graph[:, 3],  '-.',color='#a8a495', linewidth=3, marker='o', markerfacecolor='yellow', markersize=12,
                     label="GOA-AViT-SNetv2 ")
            plt.plot(learnper, Graph[:, 4], '-.', color='#004577', linewidth=3, marker='o', markerfacecolor='white', markersize=12,
                     label="MRFGO-AViT-SNetv2 ")
            plt.xticks(learnper, ('4', '8', '16', '32', '48'))
            plt.xlabel('Batch Size')
            plt.ylabel(Terms[Graph_Terms[j]])
            # plt.legend(loc=4)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            path1 = "./Results/Dataset_%s_%s_line_batch.png" % (str(a+1),Terms[Graph_Terms[j]])
            plt.savefig(path1)
            plt.show()

def Met():
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Term = [0, 1,2,3,4,5,6,7,8,9]
    for a in range(2):
        eval = np.load('Eval_all.npy', allow_pickle=True)[a]
        # for i in range(eval.shape[0]):
        for j in range(len(Graph_Term)):
            Graph = np.zeros((eval.shape[0], eval.shape[1]))
            for k in range(eval.shape[0]):
                for l in range(eval.shape[1]):
                    if j == 9:
                        Graph[k, l] = eval[k, l, Graph_Term[4] + 4]
                    else:
                        Graph[k, l] = eval[k, l, Graph_Term[j] + 4] * 100
            fig = plt.figure()
            ax = fig.add_axes([0.12, 0.1, 0.7, 0.8])
            X = np.arange(5)
            ax.bar(X + 0.00, Graph[:, 0], color='#f97306', edgecolor='k', width=0.10, hatch="+", label="CNN")
            ax.bar(X + 0.10, Graph[:, 1], color='#f10c45', edgecolor='k', width=0.10, hatch="x", label="DENSENET")
            ax.bar(X + 0.20, Graph[:, 2], color='#ddd618', edgecolor='k', width=0.10, hatch="/", label="RAN")
            ax.bar(X + 0.30, Graph[:, 3], color='#6ba353', edgecolor='k', width=0.10, hatch="o", label="ViT_SNetv2")
            ax.bar(X + 0.40, Graph[:, 4], color='#13bbaf', edgecolor='r', width=0.10, hatch="*", label="MRFGO-AViT-SNetv2")
            plt.xticks(X + 0.25, ('4', '8', '16', '32', '48'), rotation=15)
            plt.ylabel(Terms[Graph_Term[j]])
            plt.xlabel('Batch Size')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            path1 = "./Results/Dataset_%s_%s_bar_batch.png" % (str(a+1), Terms[j])
            plt.savefig(path1)
            plt.show()

def Confusion_matrix():
    for a in range(2):
        Eval = np.load('Eval_all.npy', allow_pickle=True)[a]
        value = Eval[4, 4, :5]
        val = np.asarray([0, 1, 1])
        data = {'y_Actual': [val.ravel()],
                'y_Predicted': [np.asarray(val).ravel()]
                }
        df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
        confusion_matrix = pd.crosstab(df['y_Actual'][0], df['y_Predicted'][0], rownames=['Actual'],
                                       colnames=['Predicted'])
        value = value.astype('int')

        confusion_matrix.values[0, 0] = value[1]
        confusion_matrix.values[0, 1] = value[3]
        confusion_matrix.values[1, 0] = value[2]
        confusion_matrix.values[1, 1] = value[0]

        sn.heatmap(confusion_matrix, annot=True).set(title='Accuracy = ' + str(Eval[4, 4, 4] * 100)[:5] + '%')
        sn.plotting_context()
        path1 = './Results/Confusions_%s.png' %(str(a+1))
        plt.savefig(path1)
        plt.show()


def Plot_Fitness():
    convs = []
    for a in range(2):
        conv = np.load('Fitness.npy', allow_pickle=True)[a]
        Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
        Algorithm = ['RSA', 'TFMOA', 'SCO', 'GOA', 'MRFGO']

        Value = np.zeros((conv.shape[0], 5))
        for j in range(conv.shape[0]):
            Value[j, 0] = np.min(conv[j, :])
            Value[j, 1] = np.max(conv[j, :])
            Value[j, 2] = np.mean(conv[j, :])
            Value[j, 3] = np.median(conv[j, :])
            Value[j, 4] = np.std(conv[j, :])

        Table = PrettyTable()
        Table.add_column("ALGORITHMS", Statistics)
        for j in range(len(Algorithm)):
            Table.add_column(Algorithm[j], Value[j, :])
        print('--------------------------------------------------Dataset_'+str(a+1)+'Statistical Analysis--------------------------------------------------')
        print(Table)

        iteration = np.arange(conv.shape[1])
        plt.plot(iteration, conv[0, :], color='r', linewidth=3, marker='>', markerfacecolor='blue', markersize=8,
                 label="RSA")
        plt.plot(iteration, conv[1, :], color='g', linewidth=3, marker='>', markerfacecolor='red', markersize=8,
                 label="TFMOA")
        plt.plot(iteration, conv[2, :], color='b', linewidth=3, marker='>', markerfacecolor='green', markersize=8,
                 label="SCO")
        plt.plot(iteration, conv[3, :], color='m', linewidth=3, marker='>', markerfacecolor='yellow', markersize=8,
                 label="GOA")
        plt.plot(iteration, conv[4, :], color='k', linewidth=3, marker='>', markerfacecolor='cyan', markersize=8,
                 label="MRFGO")
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        path1 = "./Results/conv_%s.png" %(str(a+1))
        plt.savefig(path1)
        plt.show()

if __name__ == '__main__':
    #
    Plot_Image_Results()
    plot_results_Batch()
    Met()
    Confusion_matrix()
    Plot_Fitness()