import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

class NN_Online(nn.Module):
    """
        Online deep learning framework based on EXP3 algorithm
    """
    def __init__(self, explore_range=4, threshold=0, patience=60000) -> None:
        super(NN_Online, self).__init__()
        self.features, self.classifiers = self._module_compose()
        self.explor_range = explore_range
        self.threshould=threshold
        self.patience=patience
        self.alpha = nn.Parameter(torch.ones(self.explor_range) / (self.explor_range), requires_grad=False)
        self.uniform_alpha = self.alpha.data.clone().detach()
        self.uniform_alpha.requires_grad = False
        self.idx_start = 0
        self.cool_down = 1
        self.current_idx = 0

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
        self.current_idx = idx
        for i, (module, classifier) in enumerate(zip(self.features, self.classifiers)):
            x = module(x)
            if i==idx:
                return classifier(x)

    def forward_train(self, x: torch.Tensor, target: torch.Tensor):
        # The input should be a batched 2D image
        assert len(x.shape)==4, "The input should be a batched 2D image"
        # Calculate the features and loss
        # idx = torch.multinomial(self.alpha*(1-self.s)+self.uniform_alpha.to(self.alpha.device)*self.s, 1, replacement=True).item() + self.idx_start
        idx = torch.multinomial(self.alpha, 1, replacement=True).item() + self.idx_start
        self.current_idx = idx
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
                # prob_biased = self.alpha.data*(1-self.s)+self.uniform_alpha.to(self.alpha.device)*self.s
                prob_biased = self.alpha.data
                g = (F.cross_entropy(pred, target) / prob_biased[i]).item() if pred is not None else 0.0
                assert (not math.isinf(g))&(not math.isnan(g)), "g not valid, with loss:{}, prob:{}, prediction:{}".format(F.cross_entropy(pred, target), prob_biased[i], pred)
                if not hasattr(self.alpha, 'alpha_acc'):
                    self.alpha.alpha_acc = self.alpha.data.clone()
                else:
                    self.alpha.data = self.alpha.data * 0.05 + (1-0.05) * self.alpha.alpha_acc.to(self.alpha.device)
                self.alpha.alpha_acc[i] = self.alpha.alpha_acc[i] * (self.beta ** g)
                # self.alpha.data[i]= self.alpha.data[i].item() * (self.beta ** g)
            # Setup a lower boundary for the alpha for the sake of numerical stability
            self.alpha.data = torch.clamp(self.alpha.data, 1e-8, None)
            self.alpha.alpha_acc = torch.clamp(self.alpha.alpha_acc, 1e-8, None)
            # Proxy of the probability
            self.alpha.alpha_acc = self.alpha.alpha_acc * (1-self.s) + self.uniform_alpha.to(self.alpha.device)*self.s
            # Normalize the alpha
            self.alpha.data = self.alpha.data.div(self.alpha.data.sum())
            self.alpha.alpha_acc = self.alpha.alpha_acc.div(self.alpha.alpha_acc.sum())
        
    def jump(self, crt, crt_type='val_acc'):
        if self.cool_down % self.patience==0:
            if crt_type=='val_acc':
                if not hasattr(self, 'val_acc'):
                    self.val_acc = crt
                if (self.val_acc*(1+self.threshould))<crt:
                    alpha_sorted = self.alpha.data.sort(descending=True)[0]
                    idx_old = self.idx_start
                    self.idx_start = torch.clamp(self.alpha.data.argmax() + self.idx_start, None, len(self.classifiers)-self.explor_range).item() if (alpha_sorted[0]/alpha_sorted[1])>3 else\
                                     torch.clamp(torch.tensor(self.explor_range-1) + self.idx_start, None, len(self.classifiers)-self.explor_range).item()
                    self.loss = crt
                    # Update the propabilities
                    if self.idx_start-idx_old>0:
                        self.alpha.data = self._prob_gen(self.alpha.data, self.idx_start-idx_old)
                        print("Now jump to:{}".format(self.idx_start))
                        self.alpha.alpha_acc = self.alpha.data.clone()
                        self.cool_down += 1
            else:
                if not hasattr(self, 'loss'):
                    self.loss = crt
                if (self.loss*(1-self.threshould))>crt:
                    # alpha_sorted = self.alpha.data.sort(descending=True)[0]
                    idx_old = self.idx_start
                    # self.idx_start = torch.clamp(self.alpha.data.argmax() + self.idx_start, None, len(self.classifiers)-self.explor_range).item() if (alpha_sorted[0]/alpha_sorted[1])>3 else\
                    #                  torch.clamp(torch.tensor(self.explor_range-1) + self.idx_start, None, len(self.classifiers)-self.explor_range).item()
                    # self.idx_start = torch.clamp(self.alpha.data.argmax() + self.idx_start, None, len(self.classifiers)-self.explor_range).item()
                    self.idx_start = torch.clamp(torch.tensor([self.current_idx]), None, len(self.classifiers)-self.explor_range).item()
                    self.loss = crt
                    # Update the propabilities
                    if self.idx_start-idx_old>0:
                        self.alpha.data = self._prob_gen(self.alpha.data, self.idx_start-idx_old)
                    # if self.idx_start-idx_old>0:
                    #     self.alpha.data=self.uniform_alpha.clone()
                        print("Now jump to:{}".format(self.idx_start))
                        self.alpha.alpha_acc = self.alpha.data.clone()
                        self.cool_down += 1
        else:
            self.cool_down += 1
    
    def _prob_gen(self, in_prob:torch.Tensor, idx:int)->torch.Tensor:
        mixed_prob = in_prob * (1-self.s) + self.uniform_alpha.to(in_prob.device)*self.s
        max_prob = mixed_prob[idx].item()
        out_prob = (1-max_prob)/(self.explor_range-1)*torch.ones_like(in_prob).detach()
        out_prob[0] = max_prob
        assert (out_prob.sum()-1.0).abs()<1e-4, "Outprob is not valid anymore:{}".format(out_prob)
        return out_prob