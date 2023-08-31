# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
parentDir = "/media/cds-1/DATA1/DLServer/Covid19SegmentationNaveen/AnamNet/COVID Segmentation Exp 1"
import sys 
sys.path.insert(0, parentDir) 
parentDir = "/media/cds-1/DATA1/DLServer/tent"
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
from unetplusplus import NestedUNet
from attenunet import AttU_Net
from enet import  ENet
from unet import UNet
from segnet import SegNet
from lednet import LEDNet
from deeplabv3plus import DeepLabV3plus
import torch.nn.functional as F
import sklearn.metrics as metrics
import seaborn as sn
import pandas  as pd
import scipy.io
from sklearn.metrics import classification_report

import os,time
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

# %%
# Load model
import torch.optim as optim
def setup_optimizer(params, lr = 1e-2):
    return optim.Adam(params,
                lr=lr,
                betas=(0.99, 0.999),
                weight_decay=0)

import tent_segm as tent
import norm
import GCE_segm 
import GCE_segm_const
import torch.nn.functional as F
from GCE_segm_const import gce, gceloss

import margin_segm

def makeMarginModel(model, steps, reference, gpuid,lamb=1):
    model = margin_segm.configure_model(model)
    params, param_names = margin_segm.collect_params(model)
    optimizer = setup_optimizer(params)
    margin_model = margin_segm.margin(model, optimizer,
                           steps=steps)
    return margin_model
def makeGCEModel(model, steps, reference, gpuid,lamb=1):
    model = GCE_segm.configure_model(model)
    params, param_names = GCE_segm.collect_params(model)
    optimizer = setup_optimizer(params)
    gce_model = GCE_segm.GCE(model, optimizer,
                           steps=steps)
    return gce_model
def makeTentModel(model, steps, reference, gpuid,lamb=1):
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(model, optimizer,
                           steps=steps)
    return tent_model
import osuda_segm
from copy import deepcopy
def makeOSUDAModel(model, modelRef, steps, reference, gpuid, lamb=1):    
    model = osuda_segm.configure_model(model)
    params, param_names = osuda_segm.collect_params(model)
    optimizer = setup_optimizer(params)
    # (self, model, modelRef, optimizer, steps=1, episodic=False, gpu=1):
    tent_model = osuda_segm.osuda(model, modelRef, optimizer,
                           steps=steps, gpu=gpuid)
    return tent_model
from  genViaSpl import gen_via_spl
import genViaSpl
def makeGenViaSplModel(model, steps, reference, gpuid, losstype,surrAnam):
    if 'AE' in losstype:
        # model = genViaSpl.configure_modelIL(model)
        model = genViaSpl.configure_model(model)
        params, param_names = genViaSpl.collect_params(model)
        # params, param_names = genViaSpl.collect_paramsIL(model)  
        print(" $$$$$$$$$$$$$$$$$$ ")  
        # for m in model.modules():
        #     if m.requires_grad_:
        #         print(m)
    else:
        model = genViaSpl.configure_model(model)
        params, param_names = genViaSpl.collect_params(model)
    optimizer = setup_optimizer(params)
#      def __init__(self, model, optimizer, reference, gpu, losstype = 'gce', steps=1, episodic=False):
    genspl_model = gen_via_spl(model=model, optimizer=optimizer, reference=reference,
                           losstype=losstype, gpu=gpuid, steps=steps, surrmodel = surrAnam )
    # for m in genspl_model.modules():
    #         if m.requires_grad_:
    #             print(m)
    # print(" Printing after initialization")
    # for name, param in genspl_model.named_parameters():
    #     print(name, param.requires_grad)
    return genspl_model

def makeGCEMConstodel(model, steps, reference, gpuid, lamb=1.0):
    model = GCE_segm_const.configure_model(model)
    params, param_names = GCE_segm.collect_params(model)
    optimizer = setup_optimizer(params)
    gce_model = GCE_segm_const.GCE(model, optimizer, reference,
                           steps=steps, gpu=gpuid, lamb=lamb)
    return gce_model
import Tentsegm_const
def makeTentConstodel(model, steps, reference, gpuid, lamb=0.2):
    model = Tentsegm_const.configure_model(model)
    params, param_names = Tentsegm_const.collect_params(model)
    optimizer = setup_optimizer(params)
    gce_model = Tentsegm_const.Tent(model, optimizer, reference,
                           steps=steps, gpu=gpuid, lamb=lamb)
    return gce_model

import Tentsegm_Div
def makeTentFRModel(model, steps, reference, gpuid, lamb=0.2):
    model = Tentsegm_Div.configure_model(model)
    params, param_names = Tentsegm_Div.collect_params(model)
    optimizer = setup_optimizer(params)
    gce_model = Tentsegm_Div.Tent(model, optimizer, reference,
                           steps=steps, gpu=gpuid, lamb=lamb)
    return gce_model

import Marginsegm_const
def makeMarginConstodel(model, steps, reference, gpuid,  lamb=1):
    model = Marginsegm_const.configure_model(model)
    params, param_names = Marginsegm_const.collect_params(model)
    optimizer = setup_optimizer(params)
    margin_model = Marginsegm_const.Margin(model, optimizer, reference,
                           steps=steps, gpu=gpuid, lamb=lamb)
    return margin_model
import LLR_segm_const
def makeLLRConstodel(model, steps, reference, gpuid, lamb=1):
    model = LLR_segm_const.configure_model(model)
    params, param_names = LLR_segm_const.collect_params(model)
    optimizer = setup_optimizer(params)
    llr_model = LLR_segm_const.llr(model, optimizer, reference,
                           steps=steps, gpu=gpuid, lamb=lamb)
    return llr_model
import TentMax_segm
def makeTentMaxModel(model, steps, reference, gpuid, lamb=1):
    model = Tentsegm_const.configure_model(model)
    params, param_names = TentMax_segm.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = TentMax_segm.TentMax(model, optimizer, reference,
                           steps=steps, gpu=gpuid, lamb=lamb)
    return tent_model
import MarginMax_segm
import GCEMax_segm
def makeMarginMaxModel(model, steps, reference, gpuid, lamb=1):
    model = MarginMax_segm.configure_model(model)
    params, param_names = MarginMax_segm.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = MarginMax_segm.MarginMax_segm(model, optimizer, reference,
                           steps=steps, gpu=gpuid, lamb=lamb)
    return tent_model
def makeGCEMaxModel(model, steps, reference, gpuid, lamb=1):
    model = GCEMax_segm.configure_model(model)
    params, param_names = GCEMax_segm.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = GCEMax_segm.GCEMax_segm(model, optimizer, reference,
                           steps=steps, gpu=gpuid, lamb=lamb)
    return tent_model

def makeNormModel(model, gpuid):
    norm_model = norm.Norm(model)
    return norm_model
from configUtils import *
def getpredLabelsAdapt(net, inpTensor, ref = None, SM = False):
    if SM == True:
        preds = net(inpTensor)
        flag = [False]
    else:
        if ref is  None:
            preds, flag = net(inpTensor)
        else:
            print("Using external reference before calling ", ref.size())           
            preds, flag = net(inpTensor, ref)
    soft = F.softmax(preds,dim=1)
    _, predLabels= torch.max(soft,dim=1)  
    return predLabels, flag

def getpredLabels(net, inpTensor):
    preds = net(inpTensor)
    soft = F.softmax(preds,dim=1)
    _, predLabels= torch.max(soft,dim=1)  
    return predLabels
    
def getpredLabelsAndLogits(net, inpTensor):
    preds = net(inpTensor)
    soft = F.softmax(preds,dim=1)
    _, predLabels= torch.max(soft,dim=1)  
    return soft.detach().cpu(), predLabels
    
def getpredLabelsAndConfidence(net, inpTensor, gpuid):
    preds = net(inpTensor)
    soft = F.softmax(preds,dim=1)
    _, predLabels= torch.max(soft,dim=1)  
    lossBase = gceloss(preds,gpuid)
    preds = preds.cpu().detach().numpy()
    return lossBase.cpu().detach().numpy(), predLabels



def getpredLabelsAndConfidenceSub(net, inpTensor, gpuid, bs=8):
    numBatches = np.int8(np.ceil(len(inpTensor) / bs))
    for k in range(numBatches):
      print(numBatches, k)
      tempx = inpTensor[k*bs:(k+1)*bs]
      temp_outputs = net(tempx).detach().cpu()
      if k ==0:
        preds = temp_outputs
      else:
        preds = torch.cat((preds ,temp_outputs),dim=0) 
    # preds = net(inpTensor)
    soft = F.softmax(preds,dim=1)
    _, predLabels= torch.max(soft,dim=1)  
    lossBase = gceloss(preds,gpuid)
    # preds = preds.cpu().detach().numpy()
    return lossBase.cpu().detach().numpy(), predLabels
def LoadBaseModel(modelFile, device):
    net = AnamNet().to(device)
    net.load_state_dict(torch.load(modelFile))    
    net.eval()
    return net
def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

def fisher_rao_logits_distance(output: torch.Tensor, target: torch.Tensor, gpu,epsilon: float = 1e-6,):
    device = torch.device('cuda:'+str(gpu if torch.cuda.is_available() else 'cpu'))
#     print(F.softmax(output,dim=1).size(), target.size())
    inner = torch.sum(
        torch.sqrt(F.softmax(output,dim=1) * target.to(device) + epsilon), dim=1
    )
    return torch.mean(2 * torch.acos(torch.clamp(inner, -1 + epsilon, 1 - epsilon)))

def train_TTASMSubj(subX, subY, subCR, net, gpuID =1, lr = 5e-4, epochs = 10):
    device = torch.device('cuda:'+str(gpuID) if torch.cuda.is_available() else 'cpu')        
    logitsBase = net(subX).detach().cpu()
    lossTrack = softmax_entropyLoss(logitsBase, gpuID)
    
    tentNet = configure_model(net)
    params, param_names = collect_params(tentNet)
    optimizer = optim.Adam(params,
                lr=lr,
                betas=(0.99, 0.999),
                weight_decay=0)
    subX = subX.to(device)
    subY = subY.to(device)
    subCR = subCR.to(device)    
    
    for epoch in (range(epochs)):
        
        subPreds = tentNet(subX)
        subLogits = F.softmax(subPreds, dim =1)
        subCRpred = returnCRpred(subLogits)
        
                    # loss1 = KLLoss_CR(subCRpred,  subCR) / len(subX)
        loss1 = fisher_rao_logits_distance(subCRpred,  subCR, gpuID) / len(subX)
        loss2 = softmax_entropyLoss(subPreds, gpuID)
        # print(subCRpred, subCR)
        print(" SME ", loss2, " Shape ", loss1)
        loss = loss1 + loss2
#         if epoch == 0:
#             lossTrack = loss1
#   
        optimizer.zero_grad()        
        loss.backward()              
        optimizer.step()   
        if epoch % 30 == 0:
            print("epoch No:", epoch)
            print(subCRpred[:2])
            print(subCR[:2])  
            print(loss1, loss2)
    print(" Loss Track ", lossTrack, " Final ", loss2)
    if loss2 >= lossTrack:
        print(" Uncertainty has not improved")
        flag = [True]
    else:
        flag = [False]
    # if loss1 > 0.12:
    #     flag = [True]
    # else:
    #     flag = [False]
    return tentNet, flag

def returnCRSlice(tempImg, numClasses):
    CR = []
    for classL in range(numClasses):
        CR.append(np.average(tempImg == classL))
 
    return CR
def returnCRSet(labelSet, numClasses = 3):
    CR_allset = []
    for k in range(len(labelSet)):
        temp = np.squeeze(labelSet[k])
        CR_allset.append(returnCRSlice(temp, numClasses))
    return CR_allset


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model
 
def KLLoss_CR(preds:torch.Tensor, targets:torch.Tensor):
    
    log_preds_prop: Tensor = (preds + 1e-10).log()

    log_target_prop: Tensor = (targets + 1e-10).log()    

    loss_prior = - torch.einsum("bc,bc->", [preds, log_target_prop])  + torch.einsum("bc,bc->", [preds, log_preds_prop])
    return loss_prior

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def softmax_entropyLoss(x: torch.Tensor, gpu):# -> torch.Tensor:
    loss = softmax_entropy(x)    
    loss = loss.mean(0)
    loss = loss.mean(0)
    loss = loss.mean(0)
    return loss

def returnCRpred(predLogits):
    for k in range(len(predLogits)):
        if k == 0:
            cr_pred = predLogits[k].mean(1).mean(1).unsqueeze(0)
        else:
            cr_pred = torch.cat((cr_pred, predLogits[k].mean(1).mean(1).unsqueeze(0)), dim = 0)
    return cr_pred

