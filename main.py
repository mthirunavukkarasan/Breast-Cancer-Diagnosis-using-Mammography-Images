import numpy as np
import os

from Image_Results import Image_plot

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import cv2 as cv
import pandas as pd
from numpy import matlib as matlib
from random import uniform
from FCM import FCM
from GOA import GOA
from Model_AViT_SNetv2 import Model_AViT_SNetv2
from Model_CNN import Model_CNN
from Model_DENSENET import Model_DENSENET
from Model_RAN import Model_RAN
from Objective_Function import Objfun_Cls
from PROPOSED import PROPOSED
from RSA import RSA
from SCO import SCO
from TFMOA import TFMOA
from Transunet import Transunet
from Transunetplusplus import Transunetplusplus
from Unet import Unet
from Unet3plus import Unet3plus
from Unet_lsa import Unet_lsa
from Plot_Results import Plot_Image_Results, Met, Confusion_matrix, Plot_Fitness, plot_results_Batch

no_of_dataset = 2

# Read Dataset
an = 0
if an == 1:
    orig = []
    gt = []
    target = []
    Directory = './Dataset/Dataset1/'
    out_folder = os.listdir(Directory)
    for j in range(len(out_folder)):
        print(j)
        if '.txt' in out_folder[j]:
            pass
        else:
            filename = Directory + out_folder[j]
            Data = cv.imread(filename)
            Data1 = cv.resize(Data, [128,128])
            image = cv.cvtColor(Data1, cv.COLOR_BGR2GRAY)
            cluster = FCM(image, image_bit=8, n_clusters=5, m=10, epsilon=0.8, max_iter=30)
            cluster.form_clusters()
            result = cluster.result.astype('uint8') * 30
            values, counts = np.unique(result, return_counts=True)
            index = np.argsort(counts)[::-1][3]
            result[result != values[index]] = 0
            analysis = cv.connectedComponentsWithStats(result, 4, cv.CV_32S)
            (totalLabels, Img, values, centroid) = analysis
            uniq, counts = np.unique(Img, return_counts=True)
            zeroIndex = np.where(uniq == 0)[0][0]
            uniq = np.delete(uniq, zeroIndex)
            counts = np.delete(counts, zeroIndex)
            sortIndex = np.argsort(counts)[::-1]
            uniq = uniq[sortIndex]
            counts = counts[sortIndex]
            remArray = []
            for j in range(len(counts)):
                if counts[j] > 500:
                    Img[Img == uniq[j]] = 0
                    remArray.append(j)
            remArray = np.array(remArray)
            if len(remArray) == 0:
                closing = np.zeros((128, 128)).astype('uint8')
                target.append(0)
                orig.append(Data1)
                gt.append(closing)
            else:
                uniq = np.delete(uniq, remArray)
                counts = np.delete(counts, remArray)
                Img[Img != 0] = 255
                Img = Img.astype('uint8')
                # Morphology Opening
                kernel = np.ones((3, 3), np.uint8)
                opening = cv.morphologyEx(Img, cv.MORPH_OPEN, kernel, iterations=1)
                # Morphology Closing
                kernel = np.ones((3, 3), np.uint8)
                closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel, iterations=1).astype('uint8')
                orig.append(Data1)
                gt.append(closing)
                target.append(1)
    np.save('Images_1.npy',orig)
    np.save('Ground_Truth_1.npy',gt)
    np.save('Target_1.npy',np.asarray(target).reshape(-1,1))

# Read Dataset 2
an = 0
if an == 1:
    orig = []
    gt = []
    target = []
    Directory = './Dataset/Dataset2/jpeg/'
    out_folder = os.listdir(Directory)
    csv_file = pd.read_csv('./Dataset/Dataset2/meta.csv')
    ind = np.where(csv_file.values[:, -1] == 1)[0]
    data = csv_file.values[ind, 0]
    for i in range(len(data)):
        print(i)
        if data[i] in out_folder:
            in_folder = Directory + data[i] + '/'
            out_folder1 = os.listdir(in_folder)
            file = in_folder + out_folder1[0]
            image = cv.imread(file)
            if image is None:
                pass
            else:
                Data1 = cv.resize(image, [128, 128])
                image = cv.cvtColor(Data1, cv.COLOR_BGR2GRAY)
                cluster = FCM(image, image_bit=8, n_clusters=5, m=10, epsilon=0.8, max_iter=30)
                cluster.form_clusters()
                result = cluster.result.astype('uint8') * 30
                values, counts = np.unique(result, return_counts=True)
                index = np.argsort(counts)[::-1][0]
                result[result != values[index]] = 0
                analysis = cv.connectedComponentsWithStats(result, 4, cv.CV_32S)
                (totalLabels, Img, values, centroid) = analysis
                uniq, counts = np.unique(Img, return_counts=True)
                zeroIndex = np.where(uniq == 0)[0][0]
                uniq = np.delete(uniq, zeroIndex)
                counts = np.delete(counts, zeroIndex)
                sortIndex = np.argsort(counts)[::-1]
                uniq = uniq[sortIndex]
                counts = counts[sortIndex]
                remArray = []
                for j in range(len(counts)):
                    if counts[j] > 500:
                        Img[Img == uniq[j]] = 0
                        remArray.append(j)
                remArray = np.array(remArray)
                if len(remArray) == 0:
                    closing = np.zeros((128, 128)).astype('uint8')
                    target.append(0)
                    orig.append(Data1)
                    gt.append(closing)
                else:
                    uniq = np.delete(uniq, remArray)
                    counts = np.delete(counts, remArray)
                    Img[Img != 0] = 255
                    Img = Img.astype('uint8')
                    # Morphology Opening
                    kernel = np.ones((3, 3), np.uint8)
                    opening = cv.morphologyEx(Img, cv.MORPH_OPEN, kernel, iterations=1)
                    # Morphology Closing
                    kernel = np.ones((3, 3), np.uint8)
                    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel, iterations=1).astype('uint8')
                    orig.append(Data1)
                    gt.append(closing)
                    target.append(1)
        else:
            pass
    np.save('Images_2.npy', orig)
    np.save('Ground_Truth_2.npy', gt)
    np.save('Target_2.npy', np.asarray(target).reshape(-1, 1))

##Unet3+_LSA Segmentation
an = 0
if an == 1:
    Segment = []
    for a in range(no_of_dataset):
        Images = np.load('Images_'+str(a+1)+'.npy', allow_pickle=True)
        GT = np.load('Ground_Truth_'+str(a+1)+'.npy', allow_pickle=True)
        per = round(Images.shape[0] * 0.75)
        train_data = Images[:per]
        train_target = GT[:per]
        test_data = Images[per:]
        test_target = GT[per:]
        Eval,Unet_Im = Unet_lsa(train_data, train_target, test_data,test_target)
        np.save('Unet_Lsa_'+str(a+1)+'.npy', Unet_Im)

# Optimization for Classification
an = 0
if an == 1:
    bestsol = []
    fitness = []
    for a in range(no_of_dataset):
        Images = np.load('Unet_Lsa_'+str(a+1)+'.npy', allow_pickle=True)
        Target = np.load('Target_'+str(a+1)+'.npy', allow_pickle=True)
        Npop = 10
        Ch_len = 4
        xmin = matlib.repmat([5,5,5,5], Npop, 1)
        xmax = matlib.repmat([255,50,255,50], Npop, 1)
        initsol = np.zeros(xmax.shape)
        for p1 in range(Npop):
            for p2 in range(xmax.shape[1]):
                initsol[p1, p2] = uniform(xmin[p1, p2], xmax[p1, p2])
        fname = Objfun_Cls
        Max_iter = 50

        print("RSA...")
        [bestfit1, fitness1, bestsol1, time1] = RSA(initsol, fname, xmin, xmax, Max_iter)

        print("TFMOA...")
        [bestfit2, fitness2, bestsol2, time2] = TFMOA(initsol, fname, xmin, xmax, Max_iter)

        print("SCO...")
        [bestfit3, fitness3, bestsol3, time3] = SCO(initsol, fname, xmin, xmax, Max_iter)

        print("GOA...")
        [bestfit4, fitness4, bestsol4, time4] = GOA(initsol, fname, xmin, xmax, Max_iter)

        print("PROPOSED...")
        [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)
        bestsol.append([bestsol1,bestsol2,bestsol3,bestsol4,bestsol5])
        fitness.append([fitness1.ravel(),fitness2.ravel(),fitness3.ravel(),fitness4.ravel(),fitness5.ravel()])
    np.save('Bestsoln.npy', np.asarray(bestsol))
    np.save('Fitneses.npy',fitness)

# Classisfication ##
an = 0
if an == 1:
    EVAL = []
    for a in range(no_of_dataset):
        Batch_Size = [4,8,16,32,48]
        Eval_all = []
        Feat = np.load('Unet_Lsa_'+str(a+1)+'.npy', allow_pickle=True)
        sol = np.load('Bestsoln.npy', allow_pickle=True)[a]
        Targets =np.load('Target_'+str(a+1)+'.npy',allow_pickle = True)
        for i in range(len(Batch_Size)):
            Eval = np.zeros((10, 14))
            learnper = round(Targets.shape[0] * Batch_Size[i])
            for j in range(sol.shape[0]):
                learnper = round(Feat.shape[0] * 0.75)
                train_data = Feat[learnper:, :]
                train_target = Targets[learnper:, :]
                test_data = Feat[:learnper, :]
                test_target = Targets[:learnper, :]
                Eval = Model_AViT_SNetv2(Feat,Targets, sol[j].astype('int'))
            Train_Data1 = Feat[learnper:, :]
            Test_Data1 = Feat[:learnper, :]
            Train_Target = Targets[learnper:, :]
            Test_Target = Targets[:learnper, :]
            Eval[5, :] = Model_CNN(Train_Data1, Train_Target, Test_Data1, Test_Target)
            Eval[6, :] = Model_DENSENET(Train_Data1, Train_Target, Test_Data1, Test_Target)
            Eval[7, :] = Model_RAN(Train_Data1, Train_Target, Test_Data1, Test_Target)
            Eval[8, :] = Model_AViT_SNetv2(Feat,Targets)
            Eval[9, :] = Eval[4, :]
            Eval_all.append(Eval)
        EVAL.append(Eval_all)
    np.save('Eval_all.npy', np.asarray(EVAL))

# Segmentation Comparison
an = 0
if an == 1:
    Eval_all = []
    for a in range(no_of_dataset):
        Images = np.load('Images_'+str(a+1)+'.npy', allow_pickle=True)
        GT = np.load('Ground_Truth_'+str(a+1)+'.npy', allow_pickle=True)
        per = round(Images.shape[0] * 0.75)
        train_data = Images[:per]
        train_target = GT[:per]
        test_data = Images[per:]
        test_target = GT[per:]
        Eval = np.zeros((5, 3))
        Eval[0 :],Image1 = Unet(train_data, train_data, test_data, test_target)
        Eval[1, :] ,Image2= Unet3plus(train_data, train_data, test_data, test_target)
        Eval[2 :],Image3 = Transunet(train_data, train_data, test_data, test_target)
        Eval[3, :],Image4 = Transunetplusplus(train_data, train_data, test_data, test_target)
        Eval[4, :],Image5 = Unet_lsa(train_data, train_data, test_data, test_target)
        np.save('Unet_'+str(a+1)+'.npy',Image1)
        np.save('Unet3+_' + str(a + 1) + '.npy', Image2)
        np.save('Transunet_' + str(a + 1) + '.npy', Image3)
        np.save('Transunetpp_' + str(a + 1) + '.npy', Image4)
        Eval_all.append(Eval)
    np.save('Eval_seg.npy',Eval_all)

Plot_Image_Results()
plot_results_Batch()
Met()
Confusion_matrix()
Plot_Fitness()
Image_plot()
