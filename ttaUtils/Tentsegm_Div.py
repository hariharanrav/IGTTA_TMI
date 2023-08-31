from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
def fisher_rao_logits_distance(
    output: torch.Tensor, target: torch.Tensor, gpu,epsilon: float = 1e-6,
):
    device = torch.device('cuda:'+str(gpu if torch.cuda.is_available() else 'cpu'))
#     print(F.softmax(output,dim=1).size(), target.size())
    inner = torch.sum(
        torch.sqrt(F.softmax(output,dim=1) * target.to(device) + epsilon), dim=1
    )
    return torch.mean(2 * torch.acos(torch.clamp(inner, -1 + epsilon, 1 - epsilon)))

# FRLoss = _InformationMeasure("kl_divergence")
def _calculate_fisher_rao_distance(preds_distribution: torch.Tensor, target_distribution: torch.Tensor) -> torch.Tensor:
        """Calculate Fisher-Rao distance between discrete distributions of predicted and reference sentences.

        Args:
            preds_distribution:
                Discrete reference distribution of predicted sentences over the vocabulary.
            target_distribution:
                Discrete reference distribution of reference sentences over the vocabulary.

        Return:
            Fisher-Rao distance between discrete distributions of predicted and reference sentences.
        """
        return 2 * torch.acos(torch.clamp(torch.sqrt(preds_distribution * target_distribution).sum(-1), 0, 1))
FRLoss = _calculate_fisher_rao_distance
# FRLoss = _InformationMeasure("kl_divergence")
from torch.nn.modules.activation import CELU
from torch.nn.modules.loss import MSELoss
import sys
parentDir = "/media/cds/storage/DATA-1/hari/tent/"
sys.path.insert(0, parentDir) 
import norm
import numpy as np
class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once Tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, reference, gpu, steps=1, lamb=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        # self.logits = logits
        # soft = F.softmax(logits,dim=1)
        # _, predLabels= torch.max(soft,dim=1)  
        # self.reference = predLabels
        self.reference = reference.cuda(gpu)
        self.lamb = lamb
        reference = reference.cpu().detach().numpy()
        # model = model.cpu().detach().numpy
        assert steps > 0, "Tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.gpu = gpu

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)


    def forward(self, x, ref = None):
        if self.episodic:
            self.reset()
        if self.steps ==1:
            outputs = self.model(x)
            return outputs, [False, 0, 0, 0]
        for count in range(self.steps):
            #print(count, self.steps)
            if ref is None:
                outputs, lossTrack = forward_and_adapt(x, self.reference, self.model, self.optimizer, self.gpu, self.lamb)        
            else:
                print("Using external reference ", x.size(), ref.size())
                outputs, lossTrack = forward_and_adapt(x, ref, self.model, self.optimizer, self.gpu, self.lamb)            
            
            '''
            if count == 0:
                # lossInit  = gceloss(self.logits,self.gpu).cpu().detach().numpy()
                lossInit = lossTrack                
                if self.steps == 1:
                    return outputsPrev, [False, lossInit, lossTrack, count]        
            
            lossPrev = lossTrack
        if lossTrack[0] > lossInit[0]:
            return outputs, [True, lossInit[0], lossTrack[0],lossPrev[0], count]
        else:
            return outputs, [False, lossInit[0], lossTrack[0],lossPrev[0], count]
            '''
            
            
     
            if count == 0:
                # lossInit  = gceloss(self.logits,self.gpu).cpu().detach().numpy()
                lossInit = lossTrack
                lossPrev = lossTrack
                outputsPrev = outputs
                if self.steps == 1:
                    return outputsPrev, [False, lossInit, lossTrack, count]        
            else:
                print(lossPrev, lossTrack)
                if sum(np.greater_equal(lossPrev, lossTrack)) == 2: #lossPrev > lossTrack:
                    print(count, "Proceeding in right direction")
                    lossPrev = lossTrack
                    outputsPrev = outputs
                    continue
                else:
                    # Check if certainity has improved
                    lossPrev = lossTrack
                    outputsPrev = outputs
                    print(count, "Divergence in one of the terms")
                    continue
        '''
        # if (lossPrev[0] >=lossInit[0]): # or (count < 3):
        
        Old
        if (count <3):
            print("\n Certainity has not improved with iterations")
            print([True, "Loss Init:", lossInit[0], "Loss Track :", lossTrack[0], "Loss Prev:" , lossPrev[0], count])
            print("\n")
            return outputsPrev, [True, lossInit[0], lossTrack[0],lossPrev[0], count]
        else:
            print("\n Yes, right")
            print([False, "Loss Init:", lossInit[0], "Loss Track :", lossTrack[0], "Loss Prev:" , lossPrev[0], count])
            print("\n")
            return outputsPrev, [False, lossInit[0], lossTrack[0],lossPrev[0], count]
            
        '''
        
        #New
        print(lossInit[0], " 1st step ", lossTrack[0], " Final step")
        #if (lossPrev[0] >=lossInit[0]):
        if (lossTrack[0] >=lossInit[0]):
            print("Certainity has not improved with iterations")
            return outputsPrev, [True, lossInit, lossPrev, count]
        elif  (count < 3):
            print("Certainity has not improved with iterations, count related")
            return outputsPrev, [True, lossInit, lossPrev, count]
        elif lossTrack[1]   > 0.12:
            print(" Rejecting the div model")
            return outputsPrev, [True, lossInit, lossPrev, count]
        else:
            return outputsPrev, [False, lossInit, lossPrev, count]

        
        # print(lossInit, lossTrack)
        # if lossTrack >= lossInit:
        #     flag = True
        # else:
        #     flag = False
        #     # outputs = self.reference
            

#         return outputs, [False, lossTrack.cpu().detach().numpy(), lossTrack.cpu().detach().numpy()]

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
#     print(x.size(), "Softmax Entropy")
#     print(torch.max(x), " ", torch.min(x), " Max and Min x")
#     s = x.softmax(1)
#     print(torch.max(s), " ", torch.min(s), s.size(), "Max and Min s")
#     print(s)
#     print(-(x.softmax(1) * x.log_softmax(1)).sum(1))
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

import torch.nn.functional as F
def gce(logits, labels, gpu,q = 0.8):
    """ Generalized cross entropy.
    
    Reference: https://arxiv.org/abs/1805.07836
    """
    # probs = torch.nn.functional.softmax(logits, dim=1)
    # print(logits.size(), labels.size())
    # print(logits.size())
    probs = logits.softmax(1)
    # print(logits.size()[1])
    # print(torch.max(labels), logits.size()[1])
    device = torch.device('cuda:'+str(gpu if torch.cuda.is_available() else 'cpu'))
    # For CPU
#     device = torch.device('cpu')
    labels1hot = F.one_hot(labels, num_classes=logits.size()[1]).to(device)    
    labels1hot = labels1hot.permute((0,3,1,2)) > 0
    probsSel = torch.masked_select(probs, labels1hot)
    loss = (1. - probsSel**q) / q
    return loss.mean()

# def gceloss(x:torch.Tensor) -> torch.Tensor:
def gceloss(x:torch.Tensor, gpu):
    logits = x
    predictions = logits.argmax(dim = 1)
    loss = gce(logits, predictions, gpu, q = 0.8)
    # print(loss.size(), loss)
    # loss = loss.mean(0)
    # loss = loss.mean(0)
    # loss = loss.mean(0)
    return loss
def softmax_entropyLoss(x: torch.Tensor,gpu) -> torch.Tensor:
    loss = softmax_entropy(x)    
    loss = loss.mean(0)
    loss = loss.mean(0)
    loss = loss.mean(0)
    return loss
def L1oneLoss(x: torch.Tensor) -> torch.Tensor:
    print(x.size())
    x = x.sum(1)
    print(x.size())
    deviceID = 'cuda:' + str(1)
    device = torch.device(deviceID if torch.cuda.is_available() else 'cpu')
    print(device)
    ones = torch.ones(x.size()).to(device)
    loss = nn.L1Loss()(ones, x) 
    return loss
def L2oneLoss(x: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(x.size())
    MSEloss = nn.MSELoss()
    loss = MSELoss(x, ones)
    return loss
def boundlossKL(outputs:torch.Tensor, predictions:torch.Tensor, gpu):
    labels = outputs.argmax(dim = 1)
    class_weights = [0.1,1,1]
    class_weights = [  1.02891531, 233.52427525,  58.85453929, 146.4245206]
    class_weights = [1, 1, 1]
#     class_weights = [  1.03332591 , 65.1320036,   64.46363079, 721.9950964 ]
    # class_weights = [ 1.36489045, 14.33264079,  5.06150599]
    device = torch.device('cuda:'+str(gpu if torch.cuda.is_available() else 'cpu'))
    #For CPU
#     device = torch.device('cpu')
    weights  = torch.FloatTensor(class_weights).to(device)
    CEloss = nn.CrossEntropyLoss(weight = weights)
    # CEloss = nn.CrossEntropyLoss()
    # print(outputs.size(), predictions.size())
    return CEloss(outputs, predictions)
#     mse = nn.MSELoss()  
#     return mse(labels.to(torch.float), predictions.to(torch.float))

# def boundloss(outputs:torch.Tensor, reference:torch.Tensor):
#     sizeO = outputs.size()
#     bl = 0
#     for k in range(sizeO[0]):
#         for l in range(sizeO[2]):
#             for m in range(sizeO[3]):
#                 vec1 = F.softmax(outputs[k,:,l,m])
#                 vec2 = reference[k,:,l,m]

#                 # print(vec1, vec2,FRLoss(vec1, vec2))
#                 bl = bl + FRLoss(vec1, vec2)
#     return bl / (sizeO[0] * sizeO[2] * sizeO[3])
@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, reference,  model, optimizer, gpu, lamb, bs = 20):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    #print("Adapting and Forwarding " )
    # forward
    #outputs = model(x)
#     print("Adapting and Forwarding " )
    # forward
    bs = len(x)
    numBatches = np.int8(np.ceil(len(x) / bs))
    for k in range(numBatches):
#       print(numBatches, k)
      tempx = x[k*bs:(k+1)*bs]
      temp_outputs = model(tempx)
      if k ==0:
        outputs = temp_outputs
      else:
        outputs = torch.cat((outputs ,temp_outputs),dim=0) 
    
    if outputs[0].size()[1] == 1:
        outputs = torch.cat((1-outputs[0], outputs[0]), dim=1)
    print(outputs.size())
    # predictions_1 = outputs.argmax(dim = 1)      
    # mse = nn.MSELoss()  
#     print(predictions_1.size(), reference.size())
#     print(mse(predictions_1.to(torch.float), reference.to(torch.float)))
    # adapt
    lossSM = softmax_entropyLoss(outputs,gpu)
    # lossConst = boundloss(outputs, reference, gpu)
    lossConst = fisher_rao_logits_distance(outputs, reference, gpu)
    # print(lossConst, lossGCE)
    # totLoss =  lossGCE + 10000 * lossConst
    totLoss =  1 * lossSM + lamb*lossConst
    print("Tent : ", lossSM, " Bound: ", lossConst, " Lambda : ", lamb, " Total: ", totLoss)
    #loss1L1 = L1oneLoss(outputs)
    #print(lossSM.size(), lossSM, loss1L1.size(), loss1L1)
    #print(lossSM.size(), " ", lossSM)
    # print(totLoss)
    # outputs = outputs.cpu().detach().numpy()
    totLoss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs, [lossSM.cpu().detach().numpy(), lossConst.cpu().detach().numpy()]


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


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with GCE."""
    # train mode, because GCE optimizes the model to minimize entropy
    
    # print(" Inside configure model /n" )
    # stats, stat_names = norm.collect_stats(model)
#     print(stat_names)
#     print((stats[1]), stats[2])
    model.train()
    # disable grad, to (re-)enable only what GCE updates
    model.requires_grad_(False)
    # configure norm for GCE updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    # stats, stat_names = norm.collect_stats(model)
#     print(stat_names)
#     print((stats[1]), stats[2])
    # print( " ########################## ")
    return model


def check_model(model):
    """Check model for compatability with GCE."""
    is_training = model.training
    assert is_training, "GCE needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "GCE needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "GCE should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "GCE needs normalization for its optimization"
