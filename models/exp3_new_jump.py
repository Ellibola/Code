import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

class NN_Online(nn.Module):
    """
        Online deep learning framework based on EXP3 algorithm
    """
    def __init__(self) -> None:
        super(NN_Online, self).__init__()
        self.features, self.classifiers = self._module_compose()
        self.idx_start = 0
        self.cool_down = 0

    def _module_compose(self)->list[list[nn.Module], list[nn.Module]]:
        raise NotImplementedError
    
    def set_hyper_params(self, beta=0.99, s=0.2, explore_range=4, threshold=1e-2, patience=120000, **kwargs):
        assert (beta<1)&(beta>0), "Invalid beta value:{}".format(beta)
        assert (s<1)&(s>0), "Invalid s value:{}".format(s)
        self.beta, self.s = beta, s
        self.threshould=threshold
        self.patience=patience
        self.explore_range = explore_range
        self.alpha = nn.Parameter(torch.ones(self.explore_range) / (self.explore_range), requires_grad=False)
        self.uniform_alpha = self.alpha.data.clone().detach()
        self.uniform_alpha.requires_grad = False
    
    def forward(self, x:torch.Tensor):
        # The input should be a batched 2D image
        assert len(x.shape)==4, "The input should be a batched 2D image"
        idx = torch.multinomial(self.alpha, 1, replacement=True).item() + self.idx_start
        for i, (module, classifier) in enumerate(zip(self.features, self.classifiers)):
            x = module(x)
            if i==idx:
                return classifier(x)

    def forward_train(self, x: torch.Tensor, target: torch.Tensor):
        # The input should be a batched 2D image
        assert len(x.shape)==4, "The input should be a batched 2D image"
        # Calculate the features and loss
        idx = torch.multinomial(self.alpha.data, 1, replacement=True).item() + self.idx_start
        prediction_list = []
        for i, (module, classifier) in enumerate(zip(self.features, self.classifiers)):
            if i<self.idx_start + self.explore_range:
                x = module(x)
                if i==idx:
                    loss = F.cross_entropy(classifier(x), target)
                    pred_final = classifier(x).detach()
                    prediction_list.append(pred_final)
                elif i>=self.idx_start:
                    prediction_list.append(None)
        return loss, pred_final, prediction_list
    
    def step(self, x: torch.Tensor, target: torch.Tensor, optimizer: torch.optim.Optimizer):
        # Update trainable parameters
        self.train()
        optimizer.zero_grad()
        loss, pred_final, prediction_list = self.forward_train(x, target)
        loss.backward()
        optimizer.step()
        # Update alpha
        self._alpha_update(prediction_list, target)
        return loss.detach(), pred_final

    def _alpha_update(self, pred_list:list[torch.Tensor], target: torch.Tensor):
        with torch.no_grad():
            assert len(self.alpha) == len(pred_list), "The length of alpha is not equal to that of the prediction list"
            for i, pred in enumerate(pred_list):
                prob = self.alpha.data.clone()
                g = (F.cross_entropy(pred, target) / prob[i]).item() if pred is not None else 0.0
                self.alpha.data[i] = self.alpha.data[i] * (self.beta ** g)
            # Setup a lower boundary for the alpha for the sake of numerical stability
            self.alpha.data = torch.clamp(self.alpha.data, 1e-8, None)
            # Proxy
            self.alpha.data = self.alpha.data * (1-self.s) + self.uniform_alpha.to(self.alpha.data.device)*self.s
            # Normalize the alpha
            self.alpha.data = self.alpha.data.div(self.alpha.data.sum())
        
    def jump(self, crt, crt_type='val_acc'):
        """
            Jump only when there is no obvious improvements after certain steps
        """
        if self.cool_down > self.patience:
            """ Jump when patience runs out """
            self.idx_start = torch.clamp(self.alpha.data[1:].argmax() + self.idx_start + 1, None, len(self.classifiers)-self.explore_range).item()
            # Reset the criterion
            if crt_type=='loss':
                self.loss = None
            else:
                self.val_acc = None
            # Update the propabilities
            self.alpha.data=self._prob_gen(self.alpha.data)
            print("Now jump to:{}".format(self.idx_start))
            # Reset cool down
            self.cool_down = 0
        else:
            if crt_type=='val_acc':
                if (not hasattr(self, 'val_acc')) or (self.val_acc==None):
                    self.val_acc = crt
                if (self.val_acc*(1+self.threshould))>=crt:
                    self.cool_down += 1
            else:
                if (not hasattr(self, 'loss')) or (self.loss==None):
                    self.loss = crt
                if (self.loss*(1-self.threshould))<=crt:
                    self.cool_down += 1
    
    def _prob_gen(self, in_prob:torch.Tensor)->torch.Tensor:
        max_prob = in_prob[1:].max()
        out_prob = (1-max_prob)/(self.explore_range-1)*torch.ones_like(in_prob).detach()
        out_prob[0] = max_prob
        assert (out_prob.sum()-1.0).abs()<1e-4, "Outprob is not valid anymore:{}".format(out_prob)
        return out_prob