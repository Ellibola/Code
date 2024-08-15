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
        self.alpha = nn.Parameter(torch.ones(len(self.classifiers)) / (len(self.classifiers)), requires_grad=False)
        self.alpha_avg = nn.Parameter(torch.ones(len(self.classifiers)) / (len(self.classifiers)), requires_grad=False)
        self.uniform_alpha = self.alpha.data.clone().detach()
        self.eta = 1.0 - 1.0 / len(self.classifiers)

    def _module_compose(self)->list[list[nn.Module], list[nn.Module]]:
        raise NotImplementedError
    
    def set_hyper_params(self, beta:float, s:float):
        assert (beta<1)&(beta>0), "Invalid beta value:{}".format(beta)
        assert (s<1)&(s>0), "Invalid s value:{}".format(s)
        self.beta, self.s = beta, s
    
    def forward(self, x:torch.Tensor):
        # The input should be a batched 2D image
        assert len(x.shape)==4, "The input should be a batched 2D image"
        idx = torch.multinomial(self.alpha_avg, 1, replacement=True).item()
        for i, (module, classifier) in enumerate(zip(self.features, self.classifiers)):
            x = module(x)
            if i==idx:
                return classifier(x)

    def forward_train(self, x: torch.Tensor, target: torch.Tensor):
        # The input should be a batched 2D image
        assert len(x.shape)==4, "The input should be a batched 2D image"
        # Calculate the features and loss
        idx = torch.multinomial(self.alpha_avg, 1, replacement=True).item()
        prediction_list = []
        for i, (module, classifier) in enumerate(zip(self.features, self.classifiers)):
            x = module(x)
            if i==idx:
                loss = F.cross_entropy(classifier(x), target)
                pred_final = classifier(x).detach()
                prediction_list.append(pred_final)
            else:
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
            self.alpha.data = self.alpha.data * 0.99 + 0.01 * self.uniform_alpha.to(self.alpha.device)
            # Update the averaged alpha
            self.alpha_avg.data = self.alpha_avg.data * self.eta + (1-self.eta) * self.alpha.data
            self.alpha_avg.data = self.alpha_avg.data.div(self.alpha.data.sum())
        assert self.alpha.sum()-1.0<1e-4, "Alpha is not valid anymore:{}, sum up to:{}".format(self.alpha, self.alpha.sum())
        assert self.alpha_avg.sum()-1.0<1e-4, "Averaged alpha is not valid anymore:{}, sum up to:{}".format(self.alpha_avg, self.alpha_avg.sum())
        