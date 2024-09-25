import torch 
import torch.nn as nn
import torch.nn.functional as F

class NN_Online(nn.Module):
    """
        Online deep learning framework based on EXP3 algorithm
    """
    def __init__(self, explore_range=4, threshold=1e-2, patience=15) -> None:
        super(NN_Online, self).__init__()
        self.features, self.classifiers = self._module_compose()
        self.explor_range = explore_range
        self.threshould=threshold
        self.patience=patience
        self.alpha = nn.Parameter(torch.ones(self.explor_range) / (self.explor_range), requires_grad=False)
        self.uniform_alpha = self.alpha.data.clone().detach()
        self.uniform_alpha.requires_grad = False
        self.idx_start = 0
        self.cool_down = 0

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
        idx = torch.multinomial(self.alpha*0.8+self.uniform_alpha.to(self.alpha.device)*0.2, 1, replacement=True).item() + self.idx_start
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
                prob_biased = self.alpha*0.8+self.uniform_alpha.to(self.alpha.device)*0.2
                g = F.cross_entropy(pred, target) / prob_biased[i] if pred is not None else 0.0
                assert g!=float('nan'), "g is not valid, with loss:{}, and prob:{}".format(F.cross_entropy(pred, target), prob_biased[i])
                if not hasattr(self.alpha, 'alpha_acc'):
                    self.alpha.alpha_acc = self.alpha.data.clone()
                else:
                    alpha_before_acc = self.alpha.data.clone()
                    self.alpha.data = self.alpha.data * 0.99 + (1-0.99) * self.alpha.alpha_acc.to(self.alpha.device)
                self.alpha.alpha_acc[i] = self.alpha.alpha_acc[i] * (self.beta ** g)
                assert (self.beta ** g)!=float('nan'), "update is not valid, with beta:{}, and g:{}".format(self.beta, g)
            # Setup a lower boundary for the alpha for the sake of numerical stability
            self.alpha.data = torch.clamp(self.alpha.data, 1e-8, None)
            self.alpha.alpha_acc = torch.clamp(self.alpha.alpha_acc, 1e-8, None)
            alpha_before_norm = self.alpha.data.clone()
            # Normalize the alpha
            self.alpha.data = self.alpha.data.div(self.alpha.data.sum())
            self.alpha.alpha_acc = self.alpha.alpha_acc.div(self.alpha.alpha_acc.sum())
        assert (self.alpha.sum()-1.0).abs()<1e-4, "Alpha is not valid anymore:{}, sum up to:{}, was {} before normalization, was {} before acc".format(self.alpha, self.alpha.sum(), alpha_before_norm, alpha_before_acc)
        
    def jump(self, crt, crt_type='val_acc'):
        if self.cool_down % self.patience==0:
            if crt_type=='val_acc':
                if not hasattr(self, 'val_acc'):
                    self.val_acc = crt
                if self.val_acc<crt:
                    alpha_sorted = self.alpha.sort(descending=True)[0]
                    self.idx_start = torch.clamp(self.alpha.argmax() + self.idx_start, None, len(self.classifiers)-self.explor_range).item() if (alpha_sorted[0]/alpha_sorted[1])>3 else\
                                     torch.clamp(torch.tensor(self.explor_range) + self.idx_start, None, len(self.classifiers)-self.explor_range).item()
                    print("Now jump to:{}".format(self.idx_start))
                    self.loss = crt
                    # Update the propabilities
                    self.alpha.data = self._prob_gen(self.alpha.data)
                    self.alpha.alpha_acc = self.alpha.data.clone()
                    self.cool_down += 1
            else:
                if not hasattr(self, 'loss'):
                    self.loss = crt
                if (self.loss*(1-self.threshould))>crt:
                    alpha_sorted = self.alpha.sort(descending=True)[0]
                    self.idx_start = torch.clamp(self.alpha.argmax() + self.idx_start, None, len(self.classifiers)-self.explor_range).item() if (alpha_sorted[0]/alpha_sorted[1])>3 else\
                                     torch.clamp(torch.tensor(self.explor_range) + self.idx_start, None, len(self.classifiers)-self.explor_range).item()
                    print("Now jump to:{}".format(self.idx_start))
                    self.loss = crt
                    # Update the propabilities
                    self.alpha.data = self._prob_gen(self.alpha.data)
                    self.alpha.alpha_acc = self.alpha.data.clone()
                    self.cool_down += 1
        else:
            self.cool_down += 1
    
    def _prob_gen(self, in_prob:torch.Tensor)->torch.Tensor:
        max_prob = in_prob.max().item()
        out_prob = (1-max_prob)/(self.explor_range-1)*torch.ones_like(in_prob).detach()
        out_prob[0] = max_prob
        return out_prob