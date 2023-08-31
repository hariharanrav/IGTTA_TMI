from copy import deepcopy
from mimetypes import suffix_map

import torch
import torch.nn as nn
import torch.jit
from torch.nn.modules.activation import CELU
from torch.nn.modules.loss import MSELoss
import sys
parentDir = "/media/cds/storage/DATA-1/hari/tent/"
sys.path.insert(0, parentDir) 
import norm
import numpy as np
def getUncertaintyLoss(losstype):
    
    if 'gce' in losstype:
        lossFn = gceloss
    elif 'ent' in losstype:
        lossFn = softmax_entropyLoss
    elif 'margin' in losstype:
        lossFn = marginLoss
    elif 'llr' in losstype:
        lossFn = llr_loss
    else:
        lossFn = gceloss
    print(" Loss type: ", losstype, " LossFn ", lossFn)
    
    return lossFn
        
class gen_via_spl(nn.Module):
    """Generalize via specialize adapts a model using various proxy loss function during testing.

    
    """
    def __init__(self, model, optimizer, reference, gpu, losstype = 'gce', steps=1, lamb= 1,episodic=False, surrmodel=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.reference = reference
        assert steps > 0, "gen_spl requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.gpu = gpu
        self.losstype = losstype
        self.lossFn = getUncertaintyLoss(losstype)
        self.lamb = lamb
        self.f_a = forward_and_adapt
        self.surrModel = surrmodel
        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)


    def forward(self, x):
        if self.episodic:
            self.reset()
        if self.steps ==1:
            outputs = self.model(x)
            return outputs, [False, 0, 0, 0]
        for count in range(self.steps):
            print(count, self.steps)
            if 'surr' in self.losstype:
                outputs, lossTrack = forward_and_adaptSurrogate(x, self.model, self.optimizer, self.gpu, self.surrModel)
            else:                
                outputs, lossTrack = forward_and_adapt(x, self.reference, self.lossFn,self.model, self.optimizer, self.gpu, self.lamb)
            
            if count == 0:                
                lossInit = lossTrack
                lossPrev = lossTrack
                outputsPrev = outputs
                if self.steps == 1:
                    if 'surr' in self.losstype:
                        return outputsPrev, [False, lossInit]
                    else:
                        return outputsPrev, [False, lossInit, lossTrack, count]        
            else:
                if 'surr' in self.losstype:
                    if lossTrack <= lossPrev:
                        print("Proceeding in right direction")
                        lossPrev = lossTrack
                        outputsPrev = outputs
                    continue
                if sum(np.greater(lossPrev, lossTrack)) == 2: #lossPrev > lossTrack:
                    print(count, "Proceeding in right direction")
                    lossPrev = lossTrack
                    outputsPrev = outputs
                    continue
                else:
                    # Check if certainity has improved
                    #lossPrev = lossTrack # to be removed           
                    #outputsPrev = outputs # to be removed         
                    print(count, "Divergence in one of the terms")
                    break # to be changed to break
        '''
        Accept or reject criterion
        '''
        if 'surr' in self.losstype:
            print(" ##############  inside Surr ")
            print(lossInit, " 1st step ", lossTrack, " Final step ", lossPrev, " Loss Prev" )
            if (lossTrack >=lossInit) or (count < 3):
                print("Certainity has not improved with iterations")
                return outputsPrev, [True, lossInit, lossTrack, count]
            else:
                print("Certianty has improved #####")
                return outputsPrev, [False, lossInit, lossTrack, count]
        else:
            print(lossInit[0], " 1st step ", lossTrack[0], " Final step")
            if (lossPrev[0] >=lossInit[0]):
                print("Certainity has not improved with iterations")
                return outputsPrev, [True, lossInit, lossTrack, count]
            elif  (count < 3):
                print("Certainity has not improved with iterations, count related")
                return outputsPrev, [True, lossInit, lossTrack, count]
            else:
                return outputsPrev, [False, lossInit, lossTrack, count]
        # print(lossInit, lossTrack)
        # if lossTrack >= lossInit:
        #     flag = True
        # else:
        #     flag = False
        #     # outputs = self.reference
            

        # return outputs, [flag, lossInit.cpu().detach().numpy(), lossTrack.cpu().detach().numpy()]

        #     '''
        #     Tracks if uncertainty and constraint loss proceeds in right direction
        #     '''
            
        #     if count == 0:
                
        #         lossInit = lossTrack
        #         lossPrev = lossTrack
        #         outputsPrev = outputs
        #         if self.steps == 1:
        #             if 'surr' in self.losstype:
        #                 return outputsPrev, [False, lossInit]
        #             else:
        #                 return outputsPrev, [False, lossInit, lossTrack, count]        
        #     else:
        #         if 'surr' in self.losstype:
        #             continue
               
        #         if sum(np.greater(lossPrev, lossTrack)) == 2: #lossPrev > lossTrack:
        #             print(count, "Proceeding in right direction")
        #             lossPrev = lossTrack
        #             outputsPrev = outputs
        #             continue
        #         else:
        #             # Check if certainity has improved
                    
        #             print(count, "Divergence in one of the terms")
        #             break
        # '''
        # Accept or reject criterion
        # '''
        # if 'surr' in self.losstype:
        #     print(lossInit, " 1st step ", lossTrack, " Final step")
        #     if (lossPrev >=lossInit) or (count < 3):
        #         "Certainity has not improved with iterations"
        #         return outputsPrev, [True, lossInit, lossTrack, count]
        #     else:
        #         return outputsPrev, [False, lossInit, lossTrack, count]
        # else:
        #     print(lossInit[0], " 1st step ", lossTrack[0], " Final step")
        #     if (lossPrev[0] >=lossInit[0]) or (count < 3):
        #         "Certainity has not improved with iterations"
        #         return outputsPrev, [True, lossInit, lossTrack, count]
        #     else:
        #         return outputsPrev, [False, lossInit, lossTrack, count]
        # '''
        # Accept or reject criterion
        # '''
        # print(lossInit[0], " 1st step ", lossTrack[0], " Final step")
        # if (lossPrev[0] >=lossInit[0]) or (count < 3):
        #     "Certainity has not improved with iterations"
        #     return outputsPrev, [True, lossInit, lossTrack, count]
        # else:
        #     return outputsPrev, [False, lossInit, lossTrack, count]
        # print(lossInit, lossTrack)
        # if lossTrack >= lossInit:
        #     flag = True
        # else:
        #     flag = False
        #     # outputs = self.reference
            

        # return outputs, [flag, lossInit.cpu().detach().numpy(), lossTrack.cpu().detach().numpy()]

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
def marginLoss(x: torch.Tensor, gpu):
    soft = x.softmax(1)
    top2 = torch.topk(soft ,2,dim=1)
    margin = top2[0][:,0,:,:] - top2[0][:,1,:,:]
    print(margin.size(), margin.mean)
    return -1 * margin.mean()
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
def llr_loss(x: torch.Tensor, gpu):
    logits = x.softmax(1)
    maxVal,_ = torch.max(logits, dim=1)
    sumVal = torch.sum(logits, dim=1) - maxVal    + 10e-5
    loss = -torch.log(torch.div(maxVal,sumVal))
    return loss.mean()
def softmax_entropyLoss(x: torch.Tensor, gpu):# -> torch.Tensor:
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


def TVLoss(x: torch.Tensor) -> torch.Tensor:
     bs_img, c_img, h_img, w_img = x.size()
     tv_h = torch.pow(x[:,:,1:,:]-x[:,:,:-1,:], 2).sum()
     tv_h = 0
     tv_w = torch.pow(x[:,:,:,1:]-x[:,:,:,:-1], 2).sum()
     return (tv_h+tv_w)/(bs_img*c_img*h_img*w_img)
def boundloss(outputs:torch.Tensor, predictions:torch.Tensor, gpu):
    labels = outputs.argmax(dim = 1)
    class_weights = [0.1,1,1]
    class_weights = [  1.02891531, 233.52427525,  58.85453929, 146.4245206]
#     class_weights = [  1.03332591 , 65.1320036,   64.46363079, 721.9950964 ]
    class_weights = [1,1,1,1,1,1,1,1,1,1,1]
    class_weights = [1,1]
    device = torch.device('cuda:'+str(gpu if torch.cuda.is_available() else 'cpu'))
    #For CPU
#     device = torch.device('cpu')
    weights  = torch.FloatTensor(class_weights).to(device)
#     CEloss = nn.CrossEntropyLoss(weight = weights)
    CEloss = nn.CrossEntropyLoss()
    print(outputs.size(), predictions.size())
    return CEloss(outputs, predictions)
#     mse = nn.MSELoss()  
#     return mse(labels.to(torch.float), predictions.to(torch.float))

# @torch.enable_grad()  # ensure grads in possible no grad context for testing
# # forward_and_adapt(x, self.reference, self.lossFn,self.model, self.optimizer, self.gpu)
# def forward_and_adaptSurrogate(x, model, optimizer, gpu, surrModel):
#     outputs = model(x)
#     soft = F.softmax(output,dim=1)
#     inpSurr =  torch.cat([x, soft], dim=1)
#     errMap = surrModel(inpSurr)
#     print(outputs.size(), soft.size(), inpSurr.size(), errMap.size())
#     totLoss = torch.sum(errMap)
#     print("totLoss: ", totLoss)
#     totLoss.backward()
#     optimizer.step()
#     optimizer.zero_grad()
#     return outputs, totLoss

@torch.enable_grad()  # ensure grads in possible no grad context for testing
# forward_and_adapt(x, self.reference, self.lossFn,self.model, self.optimizer, self.gpu)
def forward_and_adaptSurrogate(x, model, optimizer, gpu, surrModel):
    output = model(x)
    soft = F.softmax(output,dim=1)
    inpSurr =  torch.cat([x, soft], dim=1)
    errMap = surrModel(inpSurr)
    print(output.size(), soft.size(), inpSurr.size(), errMap.size())
    totLoss = torch.mean(errMap)
    print("totLoss: ", totLoss)
    totLoss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return output.cpu().detach(), totLoss.cpu().detach().numpy()
        
def forward_and_adapt(x, reference,  lossFn, model, optimizer, gpu, lamb, bs = 24):
    """
    Specializes for every sample
    """
    #print("Adapting and Forwarding " )
    # forward
    numBatches = 
    outputs = model(x)
    
    predictions_1 = outputs.argmax(dim = 1)      
    mse = nn.MSELoss()  
#     print(predictions_1.size(), reference.size())
#     print(mse(predictions_1.to(torch.float), reference.to(torch.float)))
    # adapt
#     lossGCE = gceloss(outputs,gpu)
    lossUncertain = lossFn(outputs, gpu)
    lossConst = boundloss(outputs, reference, gpu)
    lossTV = TVLoss(outputs)
    # print(lossConst, lossGCE)
    # totLoss =  lossGCE + 10000 * lossConst
#     totLoss =  1 * lossUncertain + 0.25*lossConst
    totLoss =  1 * lossUncertain + lamb*lossConst + 0.0 * lossTV
    print("UnCertainty : ", " LossFn", lossFn, " ",lossUncertain, " Bound: ", lossConst, "TV Loss", lossTV, " Lambda: ", lamb) #, "Total: ", totLoss)
    #loss1L1 = L1oneLoss(outputs)
    #print(lossSM.size(), lossSM, loss1L1.size(), loss1L1)
    #print(lossSM.size(), " ", lossSM)
    # print(totLoss)
    totLoss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs, [lossUncertain.cpu().detach().numpy(), lossConst.cpu().detach().numpy()]


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
    """Configure model for generalization."""
    
    
    # print(" Inside configure model /n" )
    # stats, stat_names = norm.collect_stats(model)
#     print(stat_names)
#     print((stats[1]), stats[2])
    model.train()
    # disable grad, to (re-)enable only what uncertainty loss updates
    model.requires_grad_(False)
    # configure norm for uncertainty model updates: enable grad + force batch statisics
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
    """Check model for compatability with gen_spl."""
    is_training = model.training
    assert is_training, "gen_spl needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "gen_spl needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "gen_spl should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "gen_spl needs normalization for its optimization"
