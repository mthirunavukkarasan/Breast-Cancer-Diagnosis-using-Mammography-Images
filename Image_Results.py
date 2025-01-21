import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def Image_plot():
    for a in range(2):
        if a == 0:
            Orig = np.load('Images_1.npy', allow_pickle=True)
            grnd = np.load('Ground_Truth_1.npy', allow_pickle=True)
            ind = [0, 1, 4, 14, 20]
        else:
            Orig = np.load('Original_Image_2.npy', allow_pickle=True)
            grnd = np.load('GT_2.npy', allow_pickle=True)
            ind = [0, 1, 2, 3, 4]
        segment = np.load('Unet_Lsa_'+str(a+1)+'.npy',allow_pickle=True)
        unet = np.load('Unet_'+str(a+1)+'.npy', allow_pickle=True)
        unet3 = np.load('Unet3+_'+str(a+1)+'.npy',allow_pickle=True)
        transunet = np.load('TransUnet_'+str(a+1)+'.npy', allow_pickle=True)
        transunetpp = np.load('Transunetpp_'+str(a+1)+'.npy', allow_pickle=True)

        for j in range(5):
            original = Orig[ind[j]]
            seg = segment[ind[j]]
            gt = grnd[ind[j]]
            un = unet[ind[j]].astype('uint8')
            un3 = unet3[ind[j]].astype('uint8')
            trn = transunet[ind[j]].astype('uint8')
            trnpp = transunetpp[ind[j]].astype('uint8')
            cv.imwrite('./Results/Image_Results/Original_'+str(j+1)+'.png',original)
            cv.imwrite('./Results/Image_Results/Grnd_' + str(j + 1) + '.png', gt)
            cv.imwrite('./Results/Image_Results/Unet_' + str(j + 1) + '.png', un)
            cv.imwrite('./Results/Image_Results/Unet3+_' + str(j + 1) + '.png', un3)
            cv.imwrite('./Results/Image_Results/Transunet_' + str(j + 1) + '.png', trn)
            cv.imwrite('./Results/Image_Results/Transunet++_' + str(j + 1) + '.png', trnpp)
            cv.imwrite('./Results/Image_Results/Proposed_' + str(j + 1) + '.png', seg)
            # fig, ax = plt.subplots(2, 2, figsize=(9, 9))
            # plt.suptitle("Image %d" % (j + 1), fontsize=20)
            # plt.subplot(1, 4, 1)
            # plt.title('Ori')
            # plt.imshow(original)
            #
            # plt.subplot(1, 4, 2)
            # plt.title('GT')
            # plt.imshow(gt)
            #
            # plt.subplot(1, 4, 3)
            # plt.title('Unet')
            # plt.imshow(un)
            #
            # plt.subplot(1, 4, 4)
            # plt.title('Unet3+')
            # plt.imshow(un3)
            #
            # plt.subplot(2, 3, 1)
            # plt.title('Transunet')
            # plt.imshow(trn)
            #
            #
            # plt.subplot(2, 3, 2)
            # plt.title('TransUnet++')
            # plt.imshow(trnpp)
            #
            # plt.subplot(2, 3, 3)
            # plt.title('Unet_Lsa')
            # plt.imshow(seg)

            # path1 = "./Results/Image_Results/disc_%s_image.png" % (str(j+1))
            # plt.savefig(path1)
            # plt.show()
if '__name__' == '__main__':
    Image_plot()