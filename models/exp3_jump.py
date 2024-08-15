import torch 
import torch.nn as nn
import torch.nn.functional as F

class NN_Online(nn.Module):
    """
        Online deep learning framework based on EXP3 algorithm
    """
    def __init__(self) -> None:
        super(NN_Online, self).__init__()
        self.features, self.classifiers = self._module_compose()
        self.explor_range = 4
        self.alpha = nn.Parameter(torch.ones(self.explor_range) / (self.explor_range), requires_grad=False)
        self.uniform_alpha = self.alpha.data.clone().detach()
        self.idx_start = 0

    def _module_compose(self)->list[list[nn.Module], list[nn.Module]]:
        raise NotImplementedError
    
    def set_hyper_params(self, beta:float, s:float):
        assert (beta<1)&(beta>0), "Invalid beta value:{}".format(beta)
        assert (s<1)&(s>0), "Invalid s value:{}".format(s)
        self.beta, self.s = beta, s
    
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
        idx = torch.multinomial(self.alpha, 1, replacement=True).item() + self.idx_start
        prediction_list = []
        for i, (module, classifier) in enumerate(zip(self.features, self.classifiers)):
            if i<self.idx_start + self.explor_range:
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
                g = F.cross_entropy(pred, target) / self.alpha.data[i] if pred is not None else 0.0
                self.alpha.data[i] = self.alpha.data[i] * (self.beta ** g)
            # Normalize the alpha
            self.alpha.data = self.alpha.data.div(self.alpha.data.sum())
            # Setup a lower boundary for the alpha
            self.alpha.data = self.alpha.data * (1-(1-self.beta)/10) + (1-self.beta)/10 * self.uniform_alpha.to(self.alpha.device)
        assert self.alpha.sum()-1.0<1e-4, "Alpha is not valid anymore:{}, sum up to:{}".format(self.alpha, self.alpha.sum())
        
    def jump(self, val_acc):
        if not hasattr(self, 'val_acc'):
            self.val_acc = val_acc
        if self.val_acc<val_acc:
            self.idx_start = torch.clamp(self.alpha.argmax() + self.idx_start, None, len(self.classifiers)-self.explor_range).item()
            print("Now jump to:{}".format(self.idx_start))
            self.val_acc = val_acc
            # Reset the propabilities
            self.alpha.data = self.uniform_alpha