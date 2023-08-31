from configUtils import *
from dataUtils import *
from predUtils import *
from modelUtils import *


import warnings
warnings.filterwarnings("ignore")
import sys
if __name__ == '__main__':

# Step1 : read input config file
    configFile = sys.argv[1]
    config = readConfig(configFile)
    
# Step2 : Load data
print(" Data Loading")
print("Experiment Id ", config.expt)
if config.expt == 1:
    
    test_data, testloader, testinp, testlab = LoadDataExpt1(config)
elif config.expt == 2:
    test_data, testloader, testinp, testlab = LoadDataExpt2(config)

# Step3 : Evaluate 
evaluationLoop(config, testloader )