from functools import partialmethod
from configUtils import * 

from predUtils import returnStats

import warnings
warnings.filterwarnings("ignore")
import sys
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg') 
import torch
import copy

def retRandVectRange(w, scale):
    r1 = -scale
    r2 = scale
    randVect = (r2 - r1) * torch.rand(w.size()) + r1 + 1
    return randVect
def perturbWeights(net, scale):
    pertNet = copy.deepcopy(net)
    for nm, m in pertNet.named_modules():
        
        if isinstance(m, nn.BatchNorm2d):
            # print(nm)
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    # print(np, p.data.size(), retRandVectRange(p.data, scale ).size(), retRandVectRange(p.data, scale ).min(), retRandVectRange(p.data, scale ).max())
                    p.data = p.data.mul_(retRandVectRange(p.data, scale ).to(device))
    pertNet.eval()
    return pertNet
if __name__ == '__main__':
    # Step1 : read input config file

    configFile = sys.argv[1]
    expt, modelFile, mode, batch_size, stepsIter, gpuid, shuffle, resultsDir, loadDataFn, LoadadaptModel, modelArch, surrmodelPath, limitedAngle, lamb = readJson(configFile)
    # Step2 : data read
    if limitedAngle:
        print("Limited Angle ", limitedAngle)
        test_data, testloader, testinp, testlab =  loadDataFn(batch_size, shuffle, limAngle = limitedAngle)
    else:
        
        test_data, testloader, testinp, testlab =  loadDataFn(batch_size, shuffle)
    device = torch.device('cuda:'+str(gpuid) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(gpuid)
    net = modelArch().to(device)
    net.load_state_dict(torch.load(modelFile))            
    net.eval()
    steps = np.array([0.1, 0.15, 0.2])
    stepsArray = []
    diffArray = []
    for step in steps:
        for k in tqdm.tqdm(range(1000)):
            pertNet = perturbWeights(net, step)
            for ind,data in tqdm.tqdm(enumerate(testloader)):             
                images,imtruth = Variable(data[0]),Variable(data[1])
                inpTensor = images.to(device)       
                
                # print("Batch number ", ind)
                
                # predBase = getpredLabels(net, inpTensor)
                lossBase, predBase = getpredLabelsAndConfidence(net,inpTensor,gpuid)
                lossPert, predPert = getpredLabelsAndConfidence(pertNet,inpTensor,gpuid)
                
                if ind==0:
                    
                    truthImages = imtruth            
                    basePredsAll=predBase
                    pertPredAll = predPert
                    inpImages = images
                else:

                    truthImages = torch.cat((truthImages ,imtruth),dim=0)            
                    basePredsAll= torch.cat((basePredsAll,predBase),dim=0)    
                    pertPredAll= torch.cat((pertPredAll,predPert),dim=0)    
                    inpImages = torch.cat((inpImages,images),dim=0)    
                    
            basePredsNP = basePredsAll.cpu().detach().numpy()
            pertPredNP = pertPredAll.cpu().detach().numpy()
            truthImagesNP = truthImages.squeeze().cpu().detach().numpy()
            AbNDiceBase, NDiceBase = returnStats(basePredsNP, truthImagesNP)
            AbNDicePert, NDicePert = returnStats(pertPredNP, truthImagesNP)
            # print(np.shape(pertPredNP), np.shape(basePredsNP), np.shape(truthImagesNP))
            print(" Base Model ", AbNDiceBase, NDiceBase)
            print(" Pert Model ", AbNDicePert, NDicePert)
            stepsArray.append(step)
            diffArray.append(AbNDicePert-AbNDiceBase)
            if k % 100 == 0:
                
                respd = pd.DataFrame()
                respd["PertbScale"] = stepsArray
                respd["DiffDice"] = diffArray
                respd.to_csv("ResultsPertAnalysis.csv")