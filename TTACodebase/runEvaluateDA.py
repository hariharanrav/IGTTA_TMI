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
    DiceAll = []
    # resultsDir = "/media/cds/storage/DATA-1/hari/Covid19SegmentationNaveen/temp/GCE_Const_BatchMode_Shuffle/"
    savefig = True
    import os
    if not os.path.exists(resultsDir):
        os.mkdir(resultsDir)
    if 'surr' in mode:
        print(" Loading surrogate, ", surrmodelPath)
        if 'AE' in mode:
            surrAnam = AnamNet(inMaps=3).cuda(gpuid)  
        else:
            surrAnam = AnamNet(inMaps=4).cuda(gpuid)  
        # surrmodelPath = '../surrogates/22Apr_1257am_surrAnam/best_model.pth'
        surrAnam.load_state_dict(torch.load(surrmodelPath))          
        # surrAnam = torch.nn.DataParallel(surrAnam, device_ids=[0, 1]).cuda()
        surrAnam = surrAnam.eval()
        surrAnam.requires_grad_(False)
    batchwiseStats = pd.DataFrame(columns = ["Batch_number", "diceABase", "diceATent", "diceNBase", "diceNTent", "FailFlag", "FgPerc", "AbPerc", "NPerc"])
    compute = True
    for ind,data in tqdm.tqdm(enumerate(testloader)):             
        images,imtruth = Variable(data[0]),Variable(data[1])
        inpTensor = images.to(device)
        
        # if ind > 0:
        #     continue
        print("Batch number ", ind)
        
        if ind == 0:
    
            net = modelArch().to(device)
            net.load_state_dict(torch.load(modelFile))    
            # net = torch.nn.DataParallel(net, device_ids=[0, 1])
            net.eval()

        
        # predBase = getpredLabels(net, inpTensor)
        lossBase, predBase = getpredLabelsAndConfidence(net,inpTensor,gpuid)
        # predBase = predBase.cpu().detach
        print("########## \n")
        print(lossBase, " : LossBase \n")
        # predLogits, predBase = getpredLabelsAndLogits(net,inpTensor)
        ### Getting logits 
        # predLogits = net(inpTensor)
        # tent_model = LoadadaptModel(net, stepsIter, predLogits,gpuid)
        if ind == 0:
            if 'surr' in mode:
                print(" Making GenViaSplModel with surrogates")
                tent_model = makeGenViaSplModel(net, stepsIter, predBase, gpuid, mode, surrAnam)
            elif 'Const' in mode:
                tent_model = LoadadaptModel(net, stepsIter, predBase,gpuid, lamb)
            else:
                tent_model = LoadadaptModel(net, stepsIter, predBase,gpuid)
        else:
            print("Resuing the tent model")
            if 'surr' in mode:
                print(" Making GenViaSplModel with surrogates")
                tent_model = makeGenViaSplModel(tent_model, stepsIter, predBase, gpuid, mode, surrAnam)
            elif 'Const' in mode:
                tent_model = LoadadaptModel(tent_model, stepsIter, predBase,gpuid, lamb)
            else:
                tent_model = LoadadaptModel(tent_model, stepsIter, predBase,gpuid)

        # try:
        predTent, flag = getpredLabelsAdapt(tent_model,inpTensor)
        predTent = predTent.cpu().detach()
        predBase = predBase.cpu().detach()
        # except:
            
        #     print("Exception")
        #     predTent = predBase
        #     flag = 1
        #     continue
            
        # del tent_model
        # print(predBase.size(), imtruth.size(), predTent.size())
        compute  = True
        if compute:
            # diceABase, diceNBase = returnStats(predBase.cpu().detach(), imtruth.cpu().detach())            
            # diceATent, diceNTent = returnStats(predTent.cpu().detach(), imtruth.cpu().detach())            
            # 
            diceABase, diceNBase = returnStats(predBase, imtruth)            
            diceATent, diceNTent = returnStats(predTent, imtruth)                
            # temp = predBase.squeeze().cpu().detach()
            temp = predBase.squeeze()
            tempArray = np.array([ind,diceABase, diceATent, diceNBase, diceNTent, flag,np.average(temp==1), np.average(temp==2), np.average(temp==0)])
            print(tempArray)
            batchwiseStats = batchwiseStats.append(pd.DataFrame(np.reshape(tempArray,(1,-1)),columns=batchwiseStats.columns), ignore_index=True)
            titleBase = "Base" + ": " + "[" + str('%.3f' % diceABase) + "," + str('%.3f' % diceNBase) + "]"
            titleTent = mode + ": " + "[" + str('%.3f' % diceATent) + "," + str('%.3f' % diceNTent) + "]"
            print("Batch Number : ", ind, " Flag", flag[0])
            print("Base" + ": " + "[" + str('%.3f' % diceABase) + "," + str('%.3f' % diceNBase) + "]")
            print(mode + ": " + "[" + str('%.3f' % diceATent) + "," + str('%.3f' % diceNTent) + "]")
        if flag[0]:
            print("Pass")
            predTent = predBase
        else:
            print(" No pass ")
        

        if ind==0:
            # truthImages = imtruth.cpu().detach()
            # tentPredsAll=predTent.cpu().detach()
            # basePredsAll=predBase.cpu().detach()
            
            truthImages = imtruth
            tentPredsAll=predTent
            basePredsAll=predBase
            inpImages = images
        else:
            # truthImages = torch.cat((truthImages ,imtruth.cpu().detach()),dim=0)
            # tentPredsAll= torch.cat((tentPredsAll,predTent.cpu().detach()),dim=0)    
            # basePredsAll= torch.cat((basePredsAll,predBase.cpu().detach()),dim=0)    
            
            truthImages = torch.cat((truthImages ,imtruth),dim=0)
            tentPredsAll= torch.cat((tentPredsAll,predTent),dim=0)    
            basePredsAll= torch.cat((basePredsAll,predBase),dim=0)    
            inpImages = torch.cat((inpImages,images),dim=0)    
        # print(imtruth.size()[0])
        compute = False
        if compute:
            for k in range(imtruth.size()[0]):
                

                plt.figure(figsize= (60,20))
                plt.subplot(141)
                plt.imshow(np.squeeze(imtruth[k]).data.cpu().numpy()); plt.title('mask', fontsize=40)#plt.colorbar(); 
                plt.subplot(142)
                plt.imshow(np.squeeze(images[k]).data.cpu().numpy(), cmap = 'gray'); plt.title('input image', fontsize=40) #plt.colorbar()
                plt.subplot(143)
                plt.imshow(np.squeeze(predBase[k]).data.cpu().numpy());  plt.title(titleBase, fontsize=40)
                plt.subplot(144)
                plt.imshow(np.squeeze(predTent[k]).data.cpu().numpy());  plt.title(titleTent, fontsize=40)
                if savefig:
    #                     plt.savefig(resultsDir + str(ind*batch_size) + "_" +str((ind+1)*batch_size) + "_" + str(master_step[steps]) + ".jpg", bbox_inches='tight',pad_inches = 0 )
                    # print(resultsDir + str(ind*batch_size + k) + "_" + str(stepsIter) + ".jpg")
                    plt.savefig(resultsDir + str(ind*batch_size + k) + ".jpg", bbox_inches='tight',pad_inches = 0 )
                    plt.close()

            print(batchwiseStats)
    tentPredsNP = tentPredsAll.numpy()
    basePredsNP = basePredsAll.numpy()
    truthImagesNP = truthImages.squeeze().numpy()
    AbNDiceBase, NDiceBase = returnStats(basePredsNP, truthImagesNP)
    AbNDiceTent, NDiceTent = returnStats(tentPredsNP, truthImagesNP)
    print(np.shape(tentPredsNP), np.shape(basePredsNP), np.shape(truthImagesNP))
    np.savez(resultsDir[:-1] + '.npz',tentPredsNP=tentPredsNP,basePredsNP=basePredsNP, truthImagesNP = truthImagesNP , inpImages=inpImages)
    print(" Base Model ", AbNDiceBase, NDiceBase)
    print(" Tent Model ", AbNDiceTent, NDiceTent)
    tempArray = np.array([ind+1,AbNDiceBase, AbNDiceTent, NDiceBase, NDiceTent, bool(False), 'NA', 'NA', 'NA'])
    batchwiseStats = batchwiseStats.append(pd.DataFrame(np.reshape(tempArray,(1,-1)),columns=batchwiseStats.columns), ignore_index=True)
    csvName = resultsDir[:-1] + '.csv'
    # print(batchwiseStats)
    print(csvName)
    batchwiseStats.to_csv(csvName)