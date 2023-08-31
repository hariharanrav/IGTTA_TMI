from functools import partialmethod
from configUtils import * 

from predUtils import returnStats

import warnings
warnings.filterwarnings("ignore")
import sys
import pandas as pd
if __name__ == '__main__':
    # Step1 : Setup Experiment
    mode = 'tentConst'
    resultsDir = resultsMainDir + "/Expt2_" + "Vol_"+ mode     
    import os
    if not os.path.exists(resultsDir):
        os.mkdir(resultsDir)
    print(resultsDir)
    # Step2 : data read
    gpuid = 1

    device = torch.device('cuda:'+str(gpuid) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(gpuid)
    DiceAll = []
    # resultsDir = "/media/cds/storage/DATA-1/hari/Covid19SegmentationNaveen/temp/GCE_Const_BatchMode_Shuffle/"
    savefig = True
    import scipy.io    
    Set4 = np.array([3,8,14,15,17])
    modelFile =  "/media/cds/storage/DATA-1/hari/Covid19SegmentationNaveen/AnamNet/COVID Segmentation Exp 2/savedModels/12Sep_0854pm_model/UNetplus_50_model.pth"
    modelArch = NestedUNet
    LoadAdaptModel = makeGCEMConstodel
    # LoadAdaptModel = makeTentConstodel
    from dataUtils import returnDataLoader
    startIndices = np.array([38,70,5,5,14]) 
    endIndices  = np.array([58,88,23, 23,34])
    for volno, k in enumerate(Set4):
        volFile = "/media/cds/storage/DATA-1/hari/Covid19SegmentationNaveen/AnamNet/COVID Segmentation Exp 2/20 CT Volume Data/patient" + str(k) + ".mat"
        dataVol = scipy.io.loadmat(volFile)
        test_data, testloader, testinp, testlab  = returnDataLoader(dataVol, startIndices[volno],endIndices[volno])
        # if (volno < 2): # or (volno == 4):
        #     continue
        # if (volno < 4):
        #     continue
        print("Volume Number :", volno)
        for ind,data in tqdm.tqdm(enumerate(testloader)):        
            images,imtruth = Variable(data[0]),Variable(data[1])
            print(ind, images.size(), imtruth.size())
            inpTensor = images.to(device)                          
    
            net = modelArch().to(device)
            net.load_state_dict(torch.load(modelFile))    
            net.eval()
            predBase = getpredLabels(net, inpTensor)
            tent_model = LoadAdaptModel(net, 10, predBase,gpuid)
            # try:
            predTent, flag = getpredLabelsAdapt(tent_model,inpTensor)
            diceABase, diceNBase = returnStats(predBase.cpu().detach(), imtruth.cpu().detach())
        
            diceATent, diceNTent = returnStats(predTent.cpu().detach(), imtruth.cpu().detach())
            if flag:
                predTent = predBase
            titleBase = "Base" + ": " + "[" + str('%.3f' % diceABase) + "," + str('%.3f' % diceNBase) + "]"
            titleTent = mode + ": " + "[" + str('%.3f' % diceATent) + "," + str('%.3f' % diceNTent) + "]"
            print("Vol Number : ", volno, " Flag", flag)
            print(titleBase)
            print(titleTent)
#     batchwiseStats = pd.DataFrame(columns = ["Batch_number", "diceABase", "diceATent", "diceNBase", "diceNTent", "FailFlag"])
#     for ind,data in tqdm.tqdm(enumerate(testloader)):             
#         images,imtruth = Variable(data[0]),Variable(data[1])
#         inpTensor = images.to(device)
        
#         # if ind > 3:
#         #     continue
#         print("Batch number ", ind)
        
    
#         net = modelArch().to(device)
#         net.load_state_dict(torch.load(modelFile))    
#         net.eval()
#         predBase = getpredLabels(net, inpTensor)
#         tent_model = LoadadaptModel(net, stepsIter, predBase,gpuid)
#         # try:
#         predTent, flag = getpredLabelsAdapt(tent_model,inpTensor)
#         # except:
            
#         #     print("Exception")
#         #     predTent = predBase
#         #     flag = 1
#         #     continue
            
#         del tent_model
#         # print(predBase.size(), imtruth.size(), predTent.size())
        
#         diceABase, diceNBase = returnStats(predBase.cpu().detach(), imtruth.cpu().detach())
        
#         diceATent, diceNTent = returnStats(predTent.cpu().detach(), imtruth.cpu().detach())
#         tempArray = np.array([ind,diceABase, diceATent, diceNBase, diceNTent, flag])
#         batchwiseStats = batchwiseStats.append(pd.DataFrame(np.reshape(tempArray,(1,-1)),columns=batchwiseStats.columns), ignore_index=True)
#         if flag:
#             predTent = predBase
#         titleBase = "Base" + ": " + "[" + str('%.3f' % diceABase) + "," + str('%.3f' % diceNBase) + "]"
#         titleTent = mode + ": " + "[" + str('%.3f' % diceATent) + "," + str('%.3f' % diceNTent) + "]"
#         print("Batch Number : ", ind, " Flag", flag)
#         print("Base" + ": " + "[" + str('%.3f' % diceABase) + "," + str('%.3f' % diceNBase) + "]")
#         print(mode + ": " + "[" + str('%.3f' % diceATent) + "," + str('%.3f' % diceNTent) + "]")

#         if ind==0:
#             truthImages = imtruth.cpu().detach()
#             tentPredsAll=predTent.cpu().detach()
#             basePredsAll=predBase.cpu().detach()
#         else:
#             truthImages = torch.cat((truthImages ,imtruth.cpu().detach()),dim=0)
#             tentPredsAll= torch.cat((tentPredsAll,predTent.cpu().detach()),dim=0)    
#             basePredsAll= torch.cat((basePredsAll,predBase.cpu().detach()),dim=0)    
#         # print(imtruth.size()[0])
#         for k in range(imtruth.size()[0]):
#             from matplotlib import pyplot as plt
#             plt.figure(figsize= (60,20))
#             plt.subplot(141)
#             plt.imshow(np.squeeze(imtruth[k]).data.cpu().numpy()); plt.title('mask', fontsize=40)#plt.colorbar(); 
#             plt.subplot(142)
#             plt.imshow(np.squeeze(images[k]).data.cpu().numpy(), cmap = 'gray'); plt.title('input image', fontsize=40) #plt.colorbar()
#             plt.subplot(143)
#             plt.imshow(np.squeeze(predBase[k]).data.cpu().numpy());  plt.title(titleBase, fontsize=40)
#             plt.subplot(144)
#             plt.imshow(np.squeeze(predTent[k]).data.cpu().numpy());  plt.title(titleTent, fontsize=40)
#             if savefig:
# #                     plt.savefig(resultsDir + str(ind*batch_size) + "_" +str((ind+1)*batch_size) + "_" + str(master_step[steps]) + ".jpg", bbox_inches='tight',pad_inches = 0 )
#                 # print(resultsDir + str(ind*batch_size + k) + "_" + str(stepsIter) + ".jpg")
#                 plt.savefig(resultsDir + str(ind*batch_size + k) + ".jpg", bbox_inches='tight',pad_inches = 0 )
#                 plt.close()

#     print(batchwiseStats)
#     tentPredsNP = tentPredsAll.numpy()
#     basePredsNP = basePredsAll.numpy()
#     truthImagesNP = truthImages.squeeze().numpy()
#     AbNDiceBase, NDiceBase = returnStats(basePredsNP, truthImagesNP)
#     AbNDiceTent, NDiceTent = returnStats(tentPredsNP, truthImagesNP)
#     print(np.shape(tentPredsNP), np.shape(basePredsNP), np.shape(truthImagesNP))
#     print(" Base Model ", AbNDiceBase, NDiceBase)
#     print(" Tent Model ", AbNDiceTent, NDiceTent)
#     tempArray = np.array([ind+1,AbNDiceBase, AbNDiceTent, NDiceBase, NDiceTent, bool(False)])
#     batchwiseStats = batchwiseStats.append(pd.DataFrame(np.reshape(tempArray,(1,-1)),columns=batchwiseStats.columns), ignore_index=True)
#     csvName = resultsDir[:-1] + '.csv'
#     print(batchwiseStats)
#     print(csvName)
#     batchwiseStats.to_csv(csvName)