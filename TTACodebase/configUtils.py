# config.resultsDir
# config.modelFile
# config.batch_size
# config.steps
# config.gpuid
# config.mode
# config.expt
# config.loadFn
# config.shuffle
import os
os.environ['CUDA_LAUNCH_BLOCKING']='1'
from dataUtils import *
from modelUtils import *
import json

class Config:
    def __init__(self, data):
        self.expt = int(data['expt'])    
        self.modelFile = data["modelFile"]
        self.mode = data["mode"]
        print(data["batch_size"])
        if "Sub" in str(data["batch_size"]):
            self.batch_size = 16
        else:
            self.batch_size = int(data["batch_size"])
        self.steps = int(data["steps"])
        self.gpuid = int(data["gpu"])
        self.shuffle = bool(data["shuffle"])
        
        NetworkName = getNetworkName(self.modelFile)
        # print(NetworkName)
        exptResultsDir = "Expt_" + str(self.expt) + "_" + NetworkName + "_"  + self.mode + "_bs_" + str(self.batch_size) + "_steps_" + str(self.steps) + "_shuffle_" + str(self.shuffle) + "/"
        self.resultsDir = os.path.join(resultsMainDir, exptResultsDir)
        print(self.resultsDir)

        if self.expt == 1:
            self.loadDataFn = LoadDataExpt1
        else:
            self.loadDataFn = LoadDataExpt2
        print("self.mode ", self.mode)
        if (self.mode == 'tent') | (self.mode == 'tentDA'):
            self.LoadadaptModel = makeTentModel
        elif (self.mode == 'norm') | (self.mode == 'normDA'):
            self.LoadadaptModel = makeNormModel
        elif self.mode == 'gce':
            self.LoadadaptModel = makeGCEModel
        elif self.mode == 'gceConst':
            self.LoadadaptModel = makeGCEMConstodel
        elif self.mode == 'surr':
            self.LoadadaptModel = makeGenViaSplModel
        elif (self.mode == 'SM_IGTTA') | (self.mode == 'SM_Anat'):
            self.LoadadaptModel = None
        

resultsMainDir = "/media/cds-1/DATA1/DLServer/Covid19SegmentationNaveen/TTAcodebase/Results_Final_SOTA/"
if not os.path.exists(resultsMainDir):
    os.mkdir(resultsMainDir)
def getNetworkName(modelFile):
    parts = modelFile.split("/")
    all = []
    for i, part in enumerate(parts):
        if "Net" in part:
            all.append(part)
            # return part
    return all[-1]
def readConfig(configFile):
    #    "expt" : "2",
    # "modelFile" : "/media/cds/storage/DATA-1/hari/Covid19SegmentationNaveen/AnamNet/COVID Segmentation Exp 2/savedModels/12Sep_0703pm_model/MiniUNet_40_model.pth", 
    # "mode"      : "gceConst",
    # "batch_size": "12",
    # "steps"     : "10",
    # "gpu"       : "1",
    # "shuffle"   : "True"
    
    with open(configFile) as f:
        data = json.load(f)
    # print(data)
    # print(int(data['expt']))
    config = Config(data)
    return config
def readJson(configFile):
    with open(configFile) as f:
        data = json.load(f)
    expt = int(data['expt'])    
    modelFile = data["modelFile"]
    mode = data["mode"]
    if "Sub" in str(data["batch_size"]):
        batch_size = str(data["batch_size"])
    else:
        batch_size = int(data["batch_size"])
    steps = int(data["steps"])
    gpuid = int(data["gpu"])
    shuffle = bool(int(data["shuffle"]))
    surrModelFile = data["surrModelFile"]
    limitedAngle = int(data["LimitedAngle"])
    lamb = float(data["lamb"])
    filters = int(data["filters"])
    print("lamb ", lamb)
    print("Printing shuffle details")
    print(shuffle, data['shuffle'])
    NetworkName = getNetworkName(modelFile)
    # print(NetworkName)
    if limitedAngle:
        exptResultsDir = "Expt_" + str(expt) + "_" + NetworkName + "_"  + mode + "_bs_" + str(batch_size) + "_steps_" + str(steps) + "_shuffle_" + str(shuffle) + "LA" + str(pow(2,limitedAngle)) + "_FixedLambda"
    else:
        exptResultsDir = "Expt_" + str(expt) + "_" + NetworkName + "_"  + mode + "_bs_" + str(batch_size) + "_steps_" + str(steps) + "_shuffle_" + str(shuffle) + "FullRecon" 
    if filters:
        exptResultsDir = "Expt_" + str(expt) + "_" + NetworkName + "_"  + mode + "_bs_" + str(batch_size) + "_steps_" + str(steps) + "_shuffle_" + str(shuffle) + "Filters_" + str(filters) 
    resultsDir = os.path.join(resultsMainDir, exptResultsDir  + "_" + str(int(lamb*10)))
    print(resultsDir)

    if expt == 1:
        if limitedAngle:
            loadDataFn = LoadDataExptLimited
        elif filters:
            loadDataFn = LoadDataExptFilters
        else:
            print(" Loading without any filters or limited angle")
            loadDataFn = LoadDataExpt1
        if "Anam" in modelFile:
            modelArch = AnamNet
        
        if "++" in modelFile:
            modelArch=NestedUNet
        if "ENet" in modelFile:
            modelArch =ENet
        
        if ("UNet" in modelFile) & ("++" not in modelFile):
            modelArch = UNet
        if "AttUNet" in modelFile:
            modelArch = AttU_Net
        if "SegNet" in modelFile:
            modelArch = SegNet 
        if "LED" in modelFile:
            modelArch = LEDNet
        if "DLV3" in modelFile:
            modelArch = DeepLabV3plus
        
        # modelArch = AttU_Net
        # modelArch = ENet
        # modelArch = NestedUNet
    elif expt ==3:
        loadDataFn = LoadDataExpt1
        modelArch = AttU_Net
    else:
        loadDataFn = LoadDataExpt2        
        modelArch = NestedUNet
        modelArch = AnamNet
    print(modelArch, " Expt ", expt)
    if (mode == 'tent') | (mode == 'tentDA'):
        LoadadaptModel = makeTentModel
    elif (mode == 'norm') | (mode == 'normDA'):
        LoadadaptModel = makeNormModel
    elif mode == 'gce':
        LoadadaptModel = makeGCEModel
    elif mode == 'gceConst':
        LoadadaptModel = makeGCEMConstodel
    elif mode == 'tentConst':
        LoadadaptModel = makeTentConstodel
    elif mode == 'marginConst':
        LoadadaptModel = makeMarginConstodel
    elif mode == 'llrConst':
        LoadadaptModel = makeLLRConstodel
    elif mode == 'tentMaxSteps':
        LoadadaptModel = makeTentMaxModel
    elif (mode == 'osuda') | (mode == 'osudaDA'):
        print("Doing OSUDA")
        LoadadaptModel = makeOSUDAModel
    elif mode == 'tentDiv':
        LoadadaptModel = makeTentFRModel
    elif 'surr' in mode:
        LoadadaptModel = makeGenViaSplModel
    elif 'SM' in mode:
        LoadadaptModel = None
    return expt, modelFile, mode, batch_size, steps, gpuid, shuffle, resultsDir, loadDataFn, LoadadaptModel, modelArch, surrModelFile, limitedAngle, lamb
import warnings
warnings.filterwarnings("ignore")
import sys
if __name__ == '__main__':
    configFile = sys.argv[1]
    config = readConfig(configFile)    