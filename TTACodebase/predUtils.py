# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
parentDir = "/media/cds/storage/DATA-1/hari/Covid19SegmentationNaveen/Segmentation-COVID-19/"
import sys 
sys.path.insert(0, parentDir) 
parentDir = "/media/cds/storage/DATA-1/hari/tent/"
sys.path.insert(0, parentDir) 

from sklearn.metrics import f1_score
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import tqdm
from config import Config
from mydataset import myDataset
import h5py as h5
from anamnet import AnamNet
import torch.nn.functional as F
import sklearn.metrics as metrics
import seaborn as sn
import pandas  as pd
import scipy.io
from sklearn.metrics import classification_report


def returnStats(tmp, tmp1):       
    gg=np.reshape(tmp1,(-1,1))     
    hh=np.reshape(tmp, (-1,1))            
          
    matrix = metrics.confusion_matrix(gg,  hh, labels = np.array([0,1,2]))
    show = 0
    if show:
        print(" confusion matrix done")
        print(matrix)
    TP = matrix[1,1]
    TN = matrix[2,2] + matrix[0,0]
    FN = matrix[1,0] + matrix[1,2]
    FP = matrix[0,1] + matrix[2,1]  
    Sens = TP / (TP+FN)
    Spec = TN / (TN+FP)
    Acc  = (TN+TP)/(TN+TP+FN+FP)
    F1sc = (2*TP) / (2*TP + FP + FN)
    F1scA = F1sc
    JIdx = F1sc / (2 - F1sc)   
    manCor = (FP + FN) / (TP + TN)
    if show:
        print("--------------------------------------------")
        print("Abnormal Class")
        print("--------------------------------------------")
        print('Sensitivity -{:.4f}' .format(Sens), 'Specificity -{:.4f}' .format(Spec), ('Accuracy    -{:.4f}' .format(Acc )))
        print('Dice Score  -{:.4f}' .format(F1sc))
        print('Manual correction Effort -{:.4f}' .format(manCor))
        # print('Jaccard Idx -{:.4f}' .format(JIdx))
        print("--------------------------------------------")
        print("Normal Class")
    TP = matrix[2,2]
    TN = matrix[1,1] + matrix[0,0]
    FN = matrix[2,0] + matrix[2,1]
    FP = matrix[0,2] + matrix[1,2]  
    Sens = TP / (TP+FN)
    Spec = TN / (TN+FP)
    Acc  = (TN+TP)/(TN+TP+FN+FP)
    F1sc = (2*TP) / (2*TP + FP + FN)
    JIdx = F1sc / (2 - F1sc)   
    F1scN = F1sc
    manCor = (FP + FN) / (TP + TN)
    if show:
        print("--------------------------------------------")
        # print('Sensitivity -{:.4f}' .format(Sens))
        # print('Specificity -{:.4f}' .format(Spec))
        # print('Accuracy    -{:.4f}' .format(Acc ))
        print('Sensitivity -{:.4f}' .format(Sens), 'Specificity -{:.4f}' .format(Spec), ('Accuracy    -{:.4f}' .format(Acc )))
        print('Dice Score  -{:.4f}' .format(F1sc))
        print('Manual correction Effort -{:.4f}' .format(manCor))
        # print('Jaccard Idx -{:.4f}' .format(JIdx))
        
    return F1scA, F1scN

# %%
from modelUtils import *
                     
def returnPredictions(config, images, imtruth,ind):
    resultsDir = config.resultsDir
    batch_size = config.batch_size
    modelFile = config.modelFile
    steps = config.steps
    mode = config.mode
    configBase = Config()
    configBase.gpuid = config.gpuid
    configBase.mode = ""
    configBase.steps = config.steps
    print("Base Model")
    baseNet = LoadModel(modelFile, "", configBase)
    outputBase = F.softmax(baseNet(images),dim=1)
    # print(output1.size())
    _, predBase= torch.max(outputBase,dim=1)  
    del baseNet        
    
    print(config.mode," Model")
    # predBase = predBase.cuda(config.gpuid)
    adaptModel = LoadModel(modelFile,predBase, config)
    outputAdapt = F.softmax(adaptModel(images),dim=1)
    # print(output.size())
    _, predAdapt= torch.max(outputAdapt,dim=1)  

    del adaptModel
    
    diceATent, diceNTent = returnStats(predAdapt.cpu().detach(), imtruth.cpu().detach())
    diceA, diceN = returnStats(predBase.cpu().detach(), imtruth.cpu().detach())
    titleBase = "Base: " + "[" + str('%.3f' % diceA) + "," + str('%.3f' % diceN) + "]"
    titleAdapt = mode + ": " + "[" + str('%.3f' % diceATent) + "," + str('%.3f' % diceNTent) + "]"
    from matplotlib import pyplot as plt
    print("Batch number : ", ind)
    print(titleBase)
    print(titleAdapt)
    for k in range(batch_size):        
        plt.figure(figsize= (60,20))
        plt.subplot(141)
        plt.imshow(np.squeeze(imtruth[k]).data.cpu().numpy()); plt.title('mask', fontsize=40)#plt.colorbar(); 
        plt.subplot(142)
        plt.imshow(np.squeeze(images[k]).data.cpu().numpy(), cmap = 'gray'); plt.title('input image', fontsize=40) #plt.colorbar()
        plt.subplot(144)
        plt.imshow(np.squeeze(predAdapt[k].data.cpu().numpy()));  plt.title(titleAdapt, fontsize=40)
        plt.subplot(143)
        plt.imshow(np.squeeze(predBase[k]).data.cpu().numpy());   plt.title(titleBase, fontsize=40)   
#   
        print(resultsDir + str(ind* batch_size + k) + "_" + str(steps) + ".jpg")
        plt.savefig(resultsDir + str(ind * batch_size + k) + "_" + str(steps) + ".jpg", bbox_inches='tight',pad_inches = 0 )
        plt.close()

    temp = predBase.squeeze().cpu().detach()
    tentPredret = predAdapt.cpu().detach()
    basePredret = predBase.cpu().detach()
    return diceATent, diceNTent, diceA, diceN, np.average(temp==1), np.average(temp==2), np.average(temp==0), tentPredret, basePredret


def evaluationLoop(config, testloader ):
       
    resultsDir = config.resultsDir 
    import os 
    diceAtentAll = []
    diceNtentAll = []
    diceAall = []
    diceNall = []
    Abmormalall = []
    Normalall = []
    Backgroundall = []
    Foregroundall = []
    if not os.path.exists(resultsDir):
        os.mkdir(resultsDir)
    for ind,data in tqdm.tqdm(enumerate(testloader)):             
            # start iterations
    #         if not i  == 38:
    #             continue
            images,imtruth = Variable(data[0]),Variable(data[1])
            # print(images.size())
            # ckeck if gpu is available
            print(" Selected gpu id during load data ", config.gpuid)
            images  = images.cuda(config.gpuid)
            imtruth = imtruth.cuda(config.gpuid)
                                                                                  
            diceATent, diceNTent, diceA, diceN, fg1, fg2, bg, tentPred, basePred = returnPredictions(config, images, imtruth, ind)
            # if (bg > 0.9) | (fg1 < 0.02) | (fg2 < 0.02) :
            #     diceAtentAll.append(diceA)
            #     diceNtentAll.append(diceN)
            #     tentPred = basePred
            # else: 
            diceAtentAll.append(diceATent)
            diceNtentAll.append(diceNTent)
            diceAall.append(diceA)        
            diceNall.append(diceN)
            Abmormalall.append(fg1)
            Normalall.append(fg2)
            Backgroundall.append(bg)
            Foregroundall.append(fg1 + fg2)
            if ind==0:
                truthImages = imtruth.cpu().detach()
                tentPreds=tentPred
                basePreds=basePred
            else:
                truthImages = torch.cat((truthImages ,imtruth.cpu().detach()),dim=0)
                tentPreds= torch.cat((tentPreds,tentPred),dim=0)    
                basePreds= torch.cat((basePreds,basePred),dim=0)    
                
            # if i >676:
            #     continue
