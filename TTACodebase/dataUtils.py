# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
parentDir = "/media/cds-1/DATA1/DLServer/Covid19SegmentationNaveen/Segmentation-COVID-19/"
import sys 
sys.path.insert(0, parentDir) 
parentDir = "/media/cds-1/DATA1/DLServer/tent/"
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

def LoadDataExpt1(batch_size, shuffle, startInd=0, endInd=704, test=True, subID = None):
    # shuffle = config.shuffle
    # batch_size = config.batch_size
    print("Experiment 1, Data Load :  Shuffle ", shuffle, " Batch_size " , batch_size)
    if test:
        dataFile = "/media/cds-1/DATA1/DLServer/Covid19SegmentationNaveen/temp/DataSet2.mat"
    else:
        dataFile = "/media/cds-1/DATA1/DLServer/Covid19SegmentationNaveen/AnamNet/COVID Segmentation Exp 1/train.mat"
    data = scipy.io.loadmat(dataFile)

    if subID is not None:
         indices = scipy.io.loadmat("/media/cds-1/DATA1/DLServer/Covid19SegmentationNaveen/indices.mat")
         subIndices = np.where(indices['idx'][0] == subID)
         startInd  = np.min(subIndices)
         endInd = np.max(subIndices)
         print(" Loading subject ", subID, " Start ", startInd, " End ", endInd)
    if test:
        inp  = data['IMG'][:,:,startInd:endInd]
        lab  = data['LAB'][:,:,startInd:endInd]
    else:
        inp  = data['inp']
        lab  = data['lab']
    
    testinp =np.reshape( np.transpose(inp,(2,0,1)),(inp.shape[2],512,512,1))
    testlab =np.reshape( np.transpose(lab,(2,0,1)),(lab.shape[2],512,512,1))

    transform = transforms.Compose([transforms.ToTensor()
                ])        

    # make the data iterator for testing data . 
    test_data = myDataset(testinp, testlab, transform)
    if subID is not None:
        testloader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=shuffle, num_workers=2)   
    else:
        testloader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=shuffle, num_workers=2)   
    return test_data, testloader, testinp, testlab

def LoadDataExptLimited(batch_size, shuffle, startInd=0, endInd=704, test=True, subID = None, limAngle = 2): # limAngle 2x = 1, 4x = 2, 8x = 3
    # shuffle = config.shuffle
    # batch_size = config.batch_size

    print("Experiment 1 Limited Angle data load, Data Load :  Shuffle ", shuffle, " Batch_size " , batch_size)
    if test:
        dataFile = "/media/cds-1/DATA1/DLServer/Covid19SegmentationNaveen/temp/DataSet2.mat"
        dataFileLimted = "/media/cds-1/DATA1/DLServer/Covid19SegmentationNaveen/temp/testVOLlimited.mat"
    else:
        dataFile = "/media/cds-1/DATA1/DLServer/Covid19SegmentationNaveen/AnamNet/COVID Segmentation Exp 1/train.mat"
    data = scipy.io.loadmat(dataFile)
    dataLimted = scipy.io.loadmat(dataFileLimted)

    if subID is not None:
         indices = scipy.io.loadmat("/media/cds-1/DATA1/DLServer/Covid19SegmentationNaveen/indices.mat")
         subIndices = np.where(indices['idx'][0] == subID)
         startInd  = np.min(subIndices)
         endInd = np.max(subIndices)
         print(" Loading subject ", subID, " Start ", startInd, " End ", endInd)
    if test:
        # inp  = data['IMG'][:,:,startInd:endInd]
        print("Loading limited angle ", limAngle)
        inp = dataLimted['testVOLlimited'][:,:,limAngle,:]
        lab  = data['LAB'][:,:,startInd:endInd]
    else:
        inp  = data['inp']
        lab  = data['lab']
    
    testinp =np.reshape( np.transpose(inp,(2,0,1)),(inp.shape[2],512,512,1))
    testlab =np.reshape( np.transpose(lab,(2,0,1)),(lab.shape[2],512,512,1))

    transform = transforms.Compose([transforms.ToTensor()
                ])        

    # make the data iterator for testing data . 
    test_data = myDataset(testinp, testlab, transform)
    if subID is not None:
        testloader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=shuffle, num_workers=2)   
    else:
        testloader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=shuffle, num_workers=2)   
    return test_data, testloader, testinp, testlab

def LoadDataExptFilters(batch_size, shuffle, startInd=0, endInd=704, test=True, subID = None, Filter = 2): # limAngle 2x = 1, 4x = 2, 8x = 3
    # shuffle = config.shuffle
    # batch_size = config.batch_size

    print("Experiment 1 Filter change data load, Data Load :  Shuffle ", shuffle, " Batch_size " , batch_size)
    if test:
        dataFile = "/media/cds-1/DATA1/DLServer/Covid19SegmentationNaveen/temp/DataSet2.mat"
        dataFileFilter = "/media/cds-1/DATA1/DLServer/Covid19SegmentationNaveen/temp/testVOLfilters.mat"
    else:
        dataFile = "/media/cds-1/DATA1/DLServer/Covid19SegmentationNaveen/AnamNet/COVID Segmentation Exp 1/train.mat"
    data = scipy.io.loadmat(dataFile)
    dataFileFilter = scipy.io.loadmat(dataFileFilter)

    if subID is not None:
         indices = scipy.io.loadmat("/media/cds-1/DATA1/DLServer/Covid19SegmentationNaveen/indices.mat")
         subIndices = np.where(indices['idx'][0] == subID)
         startInd  = np.min(subIndices)
         endInd = np.max(subIndices)
         print(" Loading subject ", subID, " Start ", startInd, " End ", endInd)
    if test:
        # inp  = data['IMG'][:,:,startInd:endInd]
        print("Loading Filter  ", Filter)
        inp = dataFileFilter['testVOLfilters'][:,:,Filter,:]
        lab  = data['LAB'][:,:,startInd:endInd]
    else:
        inp  = data['inp']
        lab  = data['lab']
    
    testinp =np.reshape( np.transpose(inp,(2,0,1)),(inp.shape[2],512,512,1))
    testlab =np.reshape( np.transpose(lab,(2,0,1)),(lab.shape[2],512,512,1))

    transform = transforms.Compose([transforms.ToTensor()
                ])        

    # make the data iterator for testing data . 
    test_data = myDataset(testinp, testlab, transform)
    if subID is not None:
        testloader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=shuffle, num_workers=2)   
    else:
        testloader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=shuffle, num_workers=2)   
    return test_data, testloader, testinp, testlab

def returnDataLoader(data, start=0, end=100):
    
    inp = data['IMG']
    lab = data['LAB']
    print(inp.shape)
    if end ==0:
        start = 0
        end = inp.shape[2]
    inp = inp[:,:,start:end]
    lab = lab[:,:,start:end]
    print(inp.shape)
    testinp =np.reshape( np.transpose(inp,(2,0,1)),(inp.shape[2],512,512,1))
    testlab =np.reshape( np.transpose(lab,(2,0,1)),(inp.shape[2],512,512,1))
    transform = transforms.Compose([transforms.ToTensor()])        
    # make the data iterator for testing data . 
    test_data = myDataset(testinp, testlab, transform)
    testloader  = torch.utils.data.DataLoader(test_data, batch_size=inp.shape[2], shuffle=False, num_workers=2)   
    return test_data, testloader, testinp, testlab   

def LoadDataExpt2(batch_size, shuffle):
    # shuffle = config.shuffle
    # batch_size = config.batch_size
    dataFile = "/media/cds/storage/DATA-1/hari/Covid19SegmentationNaveen/temp/Dataset3_set4.mat"
    print("Experiment 2, Data Load :  Shuffle ", shuffle, " Batch_size " , batch_size)
    data = scipy.io.loadmat(dataFile)
    inp  = data['inp']
    lab  = data['lab']
    
    testinp =np.reshape( np.transpose(inp,(2,0,1)),(545,512,512,1))
    testlab =np.reshape( np.transpose(lab,(2,0,1)),(545,512,512,1))

    transform = transforms.Compose([transforms.ToTensor()
                ])        

    # make the data iterator for testing data . 
    test_data = myDataset(testinp, testlab, transform)
    testloader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=shuffle, num_workers=2)   
    return test_data, testloader, testinp, testlab   
