from functools import partialmethod
from genericpath import exists
from operator import add
from unittest import result
from configUtils import * 
from predUtils import returnStats
import warnings
warnings.filterwarnings("ignore")
import sys
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import torch.nn as nn
matplotlib.use('Agg') 
from torch.utils.data import Dataset

def vizSurrPred(surrAnam, dataLoader, resultsDir, epoch, gpuid = 0):
    surrAnam.eval()
    resultsDir = os.path.join(resultsDir, "SurrPredViz" + str(epoch))
    print(resultsDir)
    os.makedirs(resultsDir, exist_ok=True)
    for i,data in tqdm.tqdm(enumerate(dataLoader)): 
        if i > 11:
            continue
        # start iterations
        images,trainLabels = Variable(data[0]),Variable(data[1])

        # ckeck if gpu is available
    #         if config.gpu == True:
        images = (images).float()  
    #         trainLabels = torch.LongTensor(trainLabels)      
        images  = images.cuda(gpuid)
        trainLabels = trainLabels.float()
        trainLabels = trainLabels.permute(0,3,1,2).cuda(gpuid)
        trainLabels = trainLabels.cpu().detach().numpy()
        output = surrAnam(images).cpu().detach().numpy()
        print(output.shape, trainLabels.shape)
        
        batch_size = len(images)
        for ind in range(batch_size):
            plt.figure(figsize=(40,40))
            plt.subplot(3,2,1); plt.imshow(output[ind, 0,:,:], cmap='gray', vmin=0, vmax=1); plt.colorbar();plt.subplot(3,2,2); plt.imshow(trainLabels[ind, 0,:,:], cmap='gray', vmin=0, vmax=1);plt.colorbar();
            plt.subplot(3,2,3); plt.imshow(output[ind, 1,:,:], cmap='gray', vmin=0, vmax=1);plt.colorbar(); plt.subplot(3,2,4); plt.imshow(trainLabels[ind, 1,:,:], cmap='gray', vmin=0, vmax=1);plt.colorbar();
            plt.subplot(3,2,5); plt.imshow(output[ind, 2,:,:], cmap='gray', vmin=0, vmax=1); plt.colorbar();plt.subplot(3,2,6); plt.imshow(trainLabels[ind, 2,:,:], cmap='gray', vmin=0, vmax=1);plt.colorbar();
            plt.savefig(resultsDir  + "/" +  str(i* batch_size + ind) + ".png")

    return 

def vizAEPred(surrAnam, dataLoader, resultsDir, epoch, gpuid = 0):
    surrAnam.eval()
    resultsDir = os.path.join(resultsDir, "AEPredViz" + str(epoch))
    print(resultsDir)
    os.makedirs(resultsDir, exist_ok=True)
    for i,data in tqdm.tqdm(enumerate(dataLoader)): 
        if i > 11:
            continue
        # start iterations
        images,trainLabels = Variable(data[0]),Variable(data[1])

        # ckeck if gpu is available
    #         if config.gpu == True:
        
        # print("Adding random masks")
        images = addRandomMasks(images)
        images = (images).float()  
#         trainLabels = torch.LongTensor(trainLabels)      
        images  = images.cuda(gpuid)
        trainLabels = torch.LongTensor(trainLabels)
        # print(trainLabels.size())
        trainLabels = trainLabels.cuda(gpuid)
        output = surrAnam(images)
        _,pred = torch.max(F.softmax(output, dim=1), dim=1)
        # print(pred.size(), trainLabels.size(), images.size())
        pred = pred.cpu().detach().numpy()
        trainLabels = trainLabels.cpu().detach().numpy()
        _,predOrig = torch.max(F.softmax(images, dim=1), dim=1)
        predOrig = predOrig.cpu().detach().numpy()
        # print(output.shape, trainLabels.shape)

        batch_size = len(images)
        for ind in range(batch_size):
            plt.figure(figsize=(15,15))
            plt.subplot(1,3,1); plt.imshow(pred[ind]) #, vmin=0, vmax=3); #plt.colorbar(); 
            plt.title('AE output')
            plt.subplot(1,3,2); plt.imshow(predOrig[ind]) #, vmin=0, vmax=3); 
            #plt.colorbar(); 
            plt.title('Base model Pred')
            plt.subplot(1,3,3); plt.imshow(trainLabels[ind]) #, vmin=0, vmax=3); 
            #plt.colorbar(); 
            plt.title('GT')           
            plt.savefig(resultsDir  + "/" +  str(i* batch_size + ind) + ".png")

    return 

class myDataset2(Dataset):
    def __init__(self, images, labels, transforms):
        self.X = images
        self.Y = labels
        self.transforms = transforms
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        datax = self.X[i, :]  
        datay = self.Y[i, :]                     
#         if self.transforms:
#             datax = self.transforms(datax).float()  
#             datay = torch.LongTensor(datay)            
        return datax, datay
def evalSurr(valloader, surrAnam):
    surrAnam.eval()
    runvalloss = 0
    criterion = nn.CrossEntropyLoss()
    for i,data in tqdm.tqdm(enumerate(valloader)): 
        # start iterations
        images,trainLabels = Variable(data[0]),Variable(data[1])
        
        # ckeck if gpu is available
#         if config.gpu == True:
        images = (images).float()  
#         trainLabels = torch.LongTensor(trainLabels)      
        images  = images.cuda(0)
        trainLabels = trainLabels.float()
        trainLabels = trainLabels.permute(0,3,1,2).cuda(0)
        output = surrAnam(images)
        loss = criterion(output,trainLabels)
        runvalloss += loss.item()
    return runvalloss

def evalAE(valloader, surrAnam, gpuid):
    surrAnam.eval()
    runvalloss = 0
    criterion = nn.CrossEntropyLoss()
    for i,data in tqdm.tqdm(enumerate(valloader)): 
        # start iterations
        images,trainLabels = Variable(data[0]),Variable(data[1])
        
        # ckeck if gpu is available
#         if config.gpu == True:
        images = (images).float()  
#         trainLabels = torch.LongTensor(trainLabels)
        # print("Adding random masks")
        images = addRandomMasks(images)      
        images  = images.cuda(gpuid)
        trainLabels = torch.LongTensor(trainLabels)
        trainLabels = trainLabels.cuda(gpuid)
        output = surrAnam(images)
        loss = criterion(output, trainLabels)
        runvalloss += loss.item()
    return runvalloss

def addRandomMasks(images, n_k = 0, size=32):
    h,w = size,size
    img = np.asarray(images.numpy())
    img_size = img.shape
    # print(img_size)
    boxes = []
    for batch in range(img_size[0]):
        for sliceID in range(img_size[1]):
            for k in range(n_k):
                y,x = np.random.randint(0,img_size[2]-w,(2,))
                img[batch, sliceID, y:y+h,x:x+w] = 0
    img = torch.Tensor(img)
    img = F.softmax(img, dim=1)
    return img
                
    
def buildAE(softF, trainLabelsF, resultsDir, testsoftF, testLabelsF, gpuid=0, epochs = 30):
    print("Number of input channels", np.shape(softF))
    print("Number of output channels", np.shape(trainLabelsF))
    surrAnam = AnamNet(inMaps=np.shape(softF)[1]).cuda(gpuid)
    optimizer = optim.Adam(surrAnam.parameters(),lr=5e-4)
    criterion = nn.CrossEntropyLoss()    
    transform = transforms.Compose([transforms.ToTensor(),
            ])
    train_data = myDataset2(np.array(softF),np.array(trainLabelsF), transform)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True, num_workers=2)
    
    test_data = myDataset2(np.array(testsoftF),np.array(testLabelsF), transform)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False, num_workers=2)
    saveDir='../surrogates/'
    from datetime import datetime
    directory=os.path.join(resultsDir , datetime.now().strftime("%d%b_%I%M%P_")+'AE_Anam_Train') 
    os.mkdir(directory)
    for j in range(10):  
        # Start epochs   
        runtrainloss = 0
        runtrainDice = 0
        surrAnam.train() 
        for i,data in tqdm.tqdm(enumerate(trainloader)): 
            # start iterations
            images,trainLabels = Variable(data[0]),Variable(data[1])
            
            # ckeck if gpu is available
    #         if config.gpu == True:
            images = (images).float()  
            print(images.size(), " Image size")
    #         trainLabels = torch.LongTensor(trainLabels)      
            # print("Adding random masks")
            images = addRandomMasks(images)
            images  = images.cuda(gpuid)
            trainLabels = torch.LongTensor(trainLabels)
            print(trainLabels.size(), " Train Labels size")
            trainLabels = trainLabels.cuda(gpuid)
            output = surrAnam(images)
            # print("Output ", output.size(), "TrainLabels, ", trainLabels.size())
            loss = criterion(output,trainLabels)
            optimizer.zero_grad()
            
            # back propagate
            loss.backward()
            
            # Accumulate loss for current minibatch
    #         dice, diceVec = dice_coef(predLabels.cpu().detach().numpy(), trainLabels.squeeze().cpu().detach().numpy())
            runtrainloss += loss.item()
    #         runtrainDice += dice
            # update the parameters
            optimizer.step()    
        runvalloss = evalAE(testloader, surrAnam, gpuid)
        print('Training - Epoch {}/{}, loss:{:.4f}'.format(j+1, 30, runtrainloss/len(trainloader)))
        
        if j%3 == 0:
            # if j == 0:
            #     continue
            torch.save(surrAnam.state_dict(),os.path.join(directory,str(j) + "_model.pth"))
            vizAEPred(surrAnam, trainloader, resultsDir,j, gpuid)   
            
        print('Validation - Epoch {}/{}, loss:{:.4f}'.format(j+1, 30, runvalloss/len(testloader)))   
    
    torch.save(surrAnam.state_dict(),os.path.join(directory,str(epochs) + "_model.pth"))
    return os.path.join(directory,str(epochs) + "_model.pth")

def buildSurrogates(inpF, errMapF, resultsDir, testinpF, testerrMapF, gpuid=0, epochs = 30):
    print("Number of input channels", np.shape(inpF))
    print("Number of output channels", np.shape(errMapF))
    surrAnam = AnamNet(inMaps=np.shape(inpF)[1]).cuda(gpuid)
    optimizer = optim.Adam(surrAnam.parameters(),lr=5e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=33, gamma=0.1)
    transform = transforms.Compose([transforms.ToTensor(),
            ])
    train_data = myDataset2(np.array(inpF),np.array(errMapF), transform)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True, num_workers=2)
    
    test_data = myDataset2(np.array(testinpF),np.array(testerrMapF), transform)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=True, num_workers=2)
    criterion = nn.MSELoss()
    for j in range(10):  
        # Start epochs   
        runtrainloss = 0
        runtrainDice = 0
        surrAnam.train() 
        for i,data in tqdm.tqdm(enumerate(trainloader)): 
            # start iterations
            images,trainLabels = Variable(data[0]),Variable(data[1])
            
            # ckeck if gpu is available
    #         if config.gpu == True:
            images = (images).float()  
    #         trainLabels = torch.LongTensor(trainLabels)      
            images  = images.cuda(0)
            trainLabels = trainLabels.float()
            # print(trainLabels.size())
            trainLabels = trainLabels.permute(0,3,1,2).cuda(0)
            output = surrAnam(images)
            print("Output ", output.size(), "TrainLabels, ", trainLabels.size())
            loss = criterion(trainLabels, output)
            optimizer.zero_grad()
            
            # back propagate
            loss.backward()
            
            # Accumulate loss for current minibatch
    #         dice, diceVec = dice_coef(predLabels.cpu().detach().numpy(), trainLabels.squeeze().cpu().detach().numpy())
            runtrainloss += loss.item()
    #         runtrainDice += dice
            # update the parameters
            optimizer.step()    
        runvalloss = evalSurr(trainloader, surrAnam)
        print('Training - Epoch {}/{}, loss:{:.4f}'.format(j+1, 30, runtrainloss/len(trainloader)))
        if j%29 == 0:
            if j == 0:
                continue
            vizSurrPred(surrAnam, testloader, resultsDir,j, gpuid)   
        # print('Validation - Epoch {}/{}, loss:{:.4f}'.format(j+1, 30, runvalloss/len(valloader)))   
    saveDir='../surrogates/'
    from datetime import datetime
    directory=os.path.join(resultsDir , datetime.now().strftime("%d%b_%I%M%P_")+'surrAnam') 
    os.mkdir(directory)
    torch.save(surrAnam.state_dict(),os.path.join(directory,str(epochs) + "_model.pth"))
    return os.path.join(directory,str(epochs) + "_model.pth")
def returnPredictions(dataLoader, modelPath, archFunc, gpuid = 0, typeR='None'):
    device = torch.device('cuda:'+str(gpuid) if  torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([transforms.ToTensor(), ])

    net=archFunc().to(device)
    net.load_state_dict(torch.load(modelPath))    
    net.eval()
    lossArray = []
    for i,data in tqdm.tqdm(enumerate(dataLoader)): 
            # if i > 0:
            #     continue
            # start iterations
            images,trainLabels = Variable(data[0]),Variable(data[1])
            # print(trainLabels.shape)
            # ckeck if gpu is available
#             if config.gpu == True:
            images  = images.cuda(device)
            trainLabels = trainLabels.cuda(device)

            # make forward pass      
            output = net(images)
            # print(images.size(), output.size(), trainLabels.size())
            soft = F.softmax(output,dim=1)   
            class_weights = [0.1,1,1]
            class_weights = torch.FloatTensor(class_weights).cuda(gpuid)
            criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
            # criterion = nn.CrossEntropyLoss()
            # print(soft.size())
            loss = torch.mean(criterion(soft,trainLabels.squeeze()), dim =[1,2])
            
#             print("InputImage: ",images.size(), " TrainLabels: ", trainLabels.squeeze().size(), "Predictions : ",output.size(), "Loss comptued: ", loss.size(),"Softmax Value: ", soft.size(), "loss value: ", loss)
            if i ==0:
                imagesArray = images.cpu().detach()
                labelsArray = trainLabels.cpu().detach()
                softArray = soft.cpu().detach()
                lossArray = loss.cpu().detach()
            else:
                imagesArray = torch.cat([imagesArray, images.cpu().detach()], dim = 0)
                labelsArray = torch.cat([labelsArray, trainLabels.cpu().detach()], dim = 0)
                softArray = torch.cat([softArray, soft.cpu().detach()], dim = 0)
                lossArray = torch.cat([lossArray, loss.cpu().detach()], dim = 0)
    print(imagesArray.size(), labelsArray.size(), softArray.size(), " Return Predictions")   
    inpArray =  torch.cat([imagesArray, softArray], dim=1)
    if 'class' in typeR:
        lossArray = lossArray.cpu().detach().numpy()
        lossDF = pd.dataFrame(columns = 'lossArray',data = lossArray)
        lossDF['bins'] =  pd.cut(lossDF['lossArray'], [0,1,3])
        
    return imagesArray.cpu().detach().numpy(), labelsArray.squeeze().cpu().detach().numpy(), softArray.cpu().detach().numpy(), inpArray.cpu().detach().numpy(), lossArray.cpu().detach().numpy()

def returnErrorMaps(labels, soft):
    print(np.shape(soft))
    numClasses = np.shape(soft)[3]
    print(numClasses)
    one_hot = F.one_hot(torch.tensor(labels).to(torch.int64), num_classes=numClasses)
    one_hotnp = one_hot.cpu().detach().numpy()
    errMap =  np.abs(one_hotnp-soft)
    return errMap

# def evalTTA_surr()
if __name__ == '__main__':
    # Step1 : read input config file
    
    configFile = sys.argv[1]
    # expt, modelFile, mode, batch_size, stepsIter, gpuid, shuffle, resultsDir, loadDataFn, LoadadaptModel, modelArch, surrModelFile = readJson(configFile)
    expt, modelFile, mode, batch_size, stepsIter, gpuid, shuffle, resultsDir, loadDataFn, LoadadaptModel, modelArch, surrmodelPath, limitedAngle, lamb = readJson(configFile)
    debug = True
    if debug:
        resultsDir = "/media/cds-1/DATA1/DLServer/Covid19SegmentationNaveen/TTAcodebase/surrogates"
        os.makedirs(resultsDir, exist_ok=True)
        resultsDir = os.path.join(resultsDir, modelFile.split('/')[-1].split('.')[0].split('_')[0])
        print(resultsDir)
        writeDir = os.path.join(resultsDir, "VizInputs")
        os.makedirs(writeDir, exist_ok=True)
    test_data, testloader, testinp, testlab = LoadDataExpt1(batch_size, shuffle, startInd=0, endInd=704)
    train_data, trainloader, traininp, trainlab = LoadDataExpt1(batch_size, False, startInd=0, endInd=704, test=False)
    imagesF, trainLabelsF, softF, inpF,lossF = returnPredictions(trainloader,modelFile,modelArch, gpuid=gpuid)
    print(imagesF.shape, trainLabelsF.shape, softF.shape, inpF.shape, lossF.shape)
    testimagesF, testLabelsF, testsoftF, testinpF,testlossF = returnPredictions(testloader,modelFile,modelArch, gpuid=gpuid)
    print(testimagesF.shape, testLabelsF.shape, testsoftF.shape, testinpF.shape, testlossF.shape)

    buildAEFlag = True
    buildSurr = False
    if buildSurr:
        print(" Preparing for neural surrogates ")
        errMapF = returnErrorMaps(trainLabelsF, softF)           
        testerrMapF = returnErrorMaps(testLabelsF, testsoftF)
        print(testerrMapF.shape)
    
    if buildSurr :
        surrModelPath = buildSurrogates(inpF, errMapF, resultsDir, testinpF, testerrMapF, gpuid)
    elif buildAEFlag:
        print(" Building AE ")
        surrModelPath = buildAE(softF, trainLabelsF, resultsDir, testsoftF, testLabelsF, gpuid)
    else: 
        surrModelPath = "/media/cds-1/DATA1/DLServer/Covid19SegmentationNaveen/TTAcodebase/surrogates/UNet/18May_0245am_surrAnam/30_model.pth"

    debug = False
    if debug:
        for k in range(len(inpF)):
            img =  np.squeeze(imagesF[k])
            gt = trainLabelsF[k]
            plt.figure(figsize=(20,20))
            plt.subplot(4,3,1); plt.imshow(img,cmap='gray')
            plt.subplot(4,3,2); plt.imshow(gt)
            plt.subplot(4,3,3); plt.imshow(gt)            
            plt.subplot(4,3,4); plt.imshow(inpF[k,1,:,:],cmap='gray')
            plt.subplot(4,3,5); plt.imshow(gt == 0,cmap='gray')
            plt.subplot(4,3,6); plt.imshow(errMapF[k,:,:,0],cmap='gray')

            plt.subplot(4,3,7); plt.imshow(inpF[k,2,:,:],cmap='gray')
            plt.subplot(4,3,8); plt.imshow(gt == 1,cmap='gray')
            plt.subplot(4,3,9); plt.imshow(errMapF[k,:,:,1],cmap='gray')

            plt.subplot(4,3,10); plt.imshow(inpF[k,3,:,:],cmap='gray')
            plt.subplot(4,3,11); plt.imshow(gt == 2,cmap='gray')
            plt.subplot(4,3,12); plt.imshow(errMapF[k,:,:,2],cmap='gray')
            plt.savefig(writeDir + "/" + str(k) + ".png")
    
    ## TestTime adaptation
    ## Load surrogate model
    surrAnam = AnamNet(inMaps=np.shape(inpF)[1]).cuda(gpuid)
    