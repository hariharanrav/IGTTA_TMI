from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
from torch.nn.modules.loss import MSELoss


class osuda(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, modelRef, gpu=1):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.gpu = gpu
        self.eta0 = 1
        self.eta = 1
        self.modelRef = modelRef

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()
        self.model.eval()
        outputs = self.model(x)
        return outputs    

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
def softmax_entropyLoss(x: torch.Tensor) -> torch.Tensor:
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
import torch.nn as nn
def get_module_by_name(module, access_string):
    names = access_string.split(sep='.')
    return reduce(getattr, names, module)
def consLoss(model, modelRef):    
    modelparams = []
    modelparamsRef = []
    MSEloss = nn.MSELoss()     
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):                           
            for nm1, m1 in modelRef.named_modules():

                if nm[6:] in nm1:
                    #print(nm,nm1)
                    for np, p in m.named_parameters():
                        modelparams.append(p.data)
                    for np1, p1 in m1.named_parameters():
                        modelparamsRef.append(p1.data)

                    
    modelparams = torch.cat(modelparams, axis=0)
    modelparamsRef = torch.cat(modelparamsRef, axis=0)
    print(modelparams.size(), modelparamsRef.size())    
             
    return MSEloss(modelparams, modelparamsRef)
    
@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, epoch, model, optimizer, model_state):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    #print("Adapting and Forwarding " )
    # forward
    outputs = model(x)
    # adapt
    loss = softmax_entropyLoss(outputs)   
    consisLoss = consLoss(model, model_state)
    print(loss, consisLoss)
    loss = loss + consisLoss
    print(loss, consisLoss)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs


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
    model_state = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


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


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
